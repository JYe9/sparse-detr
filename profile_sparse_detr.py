"""
Profile Sparse DETR (rho=0.4, R50) with the SAME protocol as
casr/scripts/profile_model.py so numbers are directly comparable on the same
hardware (V100 32GB, cuda:1 via CUDA_VISIBLE_DEVICES=1):

  FLOPs   = fvcore FlopCountAnalysis (Linear/Conv/etc.)
          + analytical supplement for the custom MSDeformAttn CUDA op, which
            fvcore cannot trace into. Per module call:
                FLOPs = Len_q x L x K x D x 10
            (Len_q queries, L=4 levels, K=4 points/head/level, D=d_model=256
            aggregated over heads; ~8 FLOPs/channel bilinear interpolation +
            2 FLOPs/channel attention-weighted accumulation per sampled
            point). Len_q is read from the ACTUAL runtime tensor shapes via
            forward hooks, so encoder sparsification (rho) is reflected
            automatically. The four nn.Linear projections inside MSDeformAttn
            are already counted by fvcore.

  Latency = CUDA events, 10 warmup + 100 timed, batch_size=1,
            torch.cuda.synchronize() each iteration.

Input: fixed 800x1333 dummy (their val transform is RandomResize([800],
max_size=1333), i.e. shorter side 800 capped at 1333 -- the canonical
DETR-family profiling resolution). Reported explicitly since it differs from
our 560x560 pipeline.
"""

import argparse
import json

import numpy as np
import torch

from fvcore.nn import FlopCountAnalysis

from main import get_args_parser
from models import build_model
from models.ops.modules import MSDeformAttn

INPUT_H, INPUT_W = 800, 1333


def build(checkpoint_path, device):
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args([
        "--with_box_refine", "--two_stage", "--eff_query_init",
        "--eff_specific_head", "--rho", "0.4", "--use_enc_aux_loss",
        "--coco_path", "data/coco",
    ])
    model, _, _ = build_model(args)
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    print(f"checkpoint loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
    assert len(missing) == 0, f"missing keys: {missing[:5]}"
    model.to(device).eval()
    return model


def main():
    device = torch.device("cuda")
    model = build("checkpoints/sparse_detr_r50_40.pth", device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.1f}M")

    x = torch.randn(1, 3, INPUT_H, INPUT_W, device=device)

    # --- MSDeformAttn analytical supplement via runtime shape hooks ---
    calls = []

    def hook(module, inputs, output):
        len_q = inputs[0].shape[1]
        calls.append(len_q * module.n_levels * module.n_points * module.d_model * 10)

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, MSDeformAttn)]
    with torch.no_grad():
        _ = model(x)
    for h in handles:
        h.remove()
    supp_flops = float(sum(calls))
    print(f"MSDeformAttn modules called: {len(calls)}, "
          f"supplement: {supp_flops/1e9:.3f} GFLOPs")

    # --- fvcore FLOPs ---
    # torch 1.10's jit tracer cannot handle the aten::fill_(Tensor, bool) op
    # inside nested_tensor_from_tensor_list (the same INTERNAL ASSERT that
    # breaks Sparse DETR's own util/benchmark.py compute_gflops on this
    # torch). Constructing the NestedTensor from plain tensor inputs inside a
    # wrapper keeps every model op in the trace while the untraceable mask
    # fill happens outside it. batch=1 with no padding => all-False mask,
    # numerically identical to the eager path.
    from util.misc import NestedTensor

    class _TraceWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tensors, mask):
            return self.m(NestedTensor(tensors, mask))

    mask = torch.zeros(1, INPUT_H, INPUT_W, dtype=torch.bool, device=device)
    wrapper = _TraceWrapper(model)
    with torch.no_grad():
        fa = FlopCountAnalysis(wrapper, (x, mask))
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        fvcore_flops = float(fa.total())
    total_flops = fvcore_flops + supp_flops
    print(f"fvcore:     {fvcore_flops/1e9:.3f} GFLOPs")
    print(f"supplement: {supp_flops/1e9:.3f} GFLOPs")
    print(f"TOTAL:      {total_flops/1e9:.3f} GFLOPs  @ {INPUT_H}x{INPUT_W}")

    # --- Latency: identical protocol to casr profile_model.py ---
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        times = []
        for _ in range(100):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = model(x)
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    lat_mean, lat_std = float(np.mean(times)), float(np.std(times))
    print(f"Latency: {lat_mean:.2f} +/- {lat_std:.2f} ms (n=100, batch=1)")

    out = {
        "model": "sparse_detr_r50_rho0.4 (official checkpoint)",
        "input_resolution": [INPUT_H, INPUT_W],
        "params_M": n_params / 1e6,
        "fvcore_gflops": fvcore_flops / 1e9,
        "msdeformattn_supplement_gflops": supp_flops / 1e9,
        "total_gflops": total_flops / 1e9,
        "latency_mean_ms": lat_mean,
        "latency_std_ms": lat_std,
        "protocol": "fvcore + analytical supplement; CUDA events 10 warmup / 100 timed, batch=1, V100",
    }
    with open("sparse_detr_profiling.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved to sparse_detr_profiling.json")


if __name__ == "__main__":
    main()
