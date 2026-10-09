#!/usr/bin/env python3
"""P4 S4: Sparse DETR (rho=0.4, R50, official checkpoint) under the P4 protocol, in the recreated `sparse-detr` env.

  --mode reproduce : the 2026-09 configuration (1x3x800x1333 torch.randn input, batch 1, 10 + 100 CUDA-event samples) to
                     check the accounted GFLOPs against casr/results/sparse_detr_profiling.json (167.344811407).
  --mode p4        : the 16 P4 profiling images (smallest COCO val2017 ids) squashed to 800x1333, ImageNet-normalised,
                     batch 1/4/16, two passes, randomised order (seed 0), GPU timeline; FP32, TF32 off, cudnn.benchmark off.
FLOPs = fvcore supported operators + the MSDeformAttn analytic supplement (Len_q x L x K x D x 10 per module call, from
runtime shapes), exactly as profile_sparse_detr.py (2026-09). Inference only; refuses to overwrite the output dir.
"""
import argparse, csv, datetime as dt, hashlib, json, os, platform, random, statistics, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import get_args_parser
from models import build_model
from models.ops.modules import MSDeformAttn
from util.misc import NestedTensor

H, W = 800, 1333
CKPT = Path(__file__).resolve().parent / "checkpoints/sparse_detr_r50_40.pth"
COCO_ANN = Path("/data/jye00001/datasets/COCO/annotations/instances_val2017.json")
COCO_IMG = Path("/data/jye00001/datasets/COCO/val2017")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa
        return "<%s failed: %s>" % (cmd[0], e)


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class GpuTimeline(threading.Thread):
    def __init__(self, period=5.0):
        super().__init__(daemon=True); self.period, self.rows, self.stop = period, [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            self.rows.append({"t": now(), "procs": run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"]).strip().splitlines(),
                              "gpus": run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active", "--format=csv,noheader"]).strip().splitlines()})
            self.stop.wait(self.period)


def build(device):
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args(["--with_box_refine", "--two_stage", "--eff_query_init", "--eff_specific_head", "--rho", "0.4", "--use_enc_aux_loss", "--coco_path", "data/coco"])
    model, _, _ = build_model(args)
    state = torch.load(str(CKPT), map_location="cpu")
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    assert len(missing) == 0, missing[:5]
    return model.to(device).eval(), {"checkpoint": str(CKPT), "checkpoint_sha256": sha256_file(CKPT), "strict_load": False, "missing_keys": list(missing), "unexpected_keys": list(unexpected),
                                     "args": "--with_box_refine --two_stage --eff_query_init --eff_specific_head --rho 0.4 --use_enc_aux_loss", "params_total": sum(p.numel() for p in model.parameters())}


class _TraceWrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, tensors, mask):
        return self.m(NestedTensor(tensors, mask))


def load_images(n, device):
    ann = json.loads(COCO_ANN.read_text()); imgs = sorted(ann["images"], key=lambda r: r["id"])[:n]
    arrs = []
    for r in imgs:
        im = Image.open(COCO_IMG / r["file_name"]).convert("RGB").resize((W, H), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = (x - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
        arrs.append(x)
    x = torch.stack(arrs).to(device); ids = [r["id"] for r in imgs]
    return x, {"image_ids": ids, "image_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "resize": "PIL bilinear squash to 800x1333 (no letterbox, mask all False), ImageNet mean/std",
               "tensor_sha256": hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest()}


def measure(model, x, warmups, samples, want_flops):
    b = int(x.shape[0]); mask = torch.zeros(b, H, W, dtype=torch.bool, device=x.device)
    out = {"batch": b, "input_shape": list(x.shape)}
    with torch.no_grad():
        calls = []
        hook = lambda module, inputs, output: calls.append(inputs[0].shape[0] * inputs[0].shape[1] * module.n_levels * module.n_points * module.d_model * 10)  # batch x Len_q x L x K x D x 10
        handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, MSDeformAttn)]
        try:
            model(NestedTensor(x, mask))
        except RuntimeError as exc:
            if "out of memory" in str(exc):
                for h in handles: h.remove()
                torch.cuda.empty_cache(); return {**out, "status": "OOM", "error": str(exc)[:300]}
            raise
        for h in handles: h.remove()
        supp = float(sum(calls)) / 1e9
        out["msdeformattn_calls"] = len(calls); out["msdeformattn_len_q"] = [int(c / (b * m.n_levels * m.n_points * m.d_model * 10)) for c, m in zip(calls, [m for m in model.modules() if isinstance(m, MSDeformAttn)])]
        if want_flops:
            fa = FlopCountAnalysis(_TraceWrapper(model), (x, mask)); fa.unsupported_ops_warnings(False); fa.uncalled_modules_warnings(False)
            supported = float(fa.total()) / 1e9; unsupported = {str(k): int(v) for k, v in fa.unsupported_ops().items()}
            del fa
            out["flops"] = {"fvcore_supported_gflops": supported, "msdeformattn_supplement_gflops": supp, "accounted_total_gflops": supported + supp, "unsupported_ops": unsupported,
                            "scope": "fvcore-supported operators + analytic MSDeformAttn term (Len_q x L x K x D x 10 per call, runtime shapes); totals for the whole batch"}
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(warmups): model(NestedTensor(x, mask))
        torch.cuda.synchronize(); t = []
        for _ in range(samples):
            torch.cuda.synchronize(); a = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            a.record(); model(NestedTensor(x, mask)); e.record(); torch.cuda.synchronize(); t.append(float(a.elapsed_time(e)))
        out["peak_cuda_memory_gb"] = {"allocated": torch.cuda.max_memory_allocated() / 1e9, "reserved": torch.cuda.max_memory_reserved() / 1e9}
    s = sorted(t)
    out["latency_ms"] = {"mean": statistics.mean(t), "sample_sd": statistics.stdev(t) if len(t) > 1 else 0.0, "median": statistics.median(t), "p5": s[int(0.05 * (len(s) - 1))], "p95": s[int(0.95 * (len(s) - 1))], "min": s[0], "max": s[-1], "n": len(t), "warmups": warmups, "samples": t}
    out["latency_per_image_ms"] = out["latency_ms"]["mean"] / b; out["throughput_images_per_s"] = 1000.0 * b / out["latency_ms"]["mean"]; out["status"] = "ok"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reproduce", "p4"], required=True); ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batches", default="1,4,16"); ap.add_argument("--warmups", type=int, default=10); ap.add_argument("--samples", type=int, default=100); ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1") or torch.cuda.device_count() != 1:
        raise SystemExit("set CUDA_VISIBLE_DEVICES to exactly one physical GPU")
    if a.output_dir.exists():
        raise SystemExit("refuse to overwrite %s" % a.output_dir)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False; torch.set_grad_enabled(False)
    a.output_dir.mkdir(parents=True)
    device = torch.device("cuda:0")
    env = {"time_start": now(), "host": platform.node(), "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu_name": torch.cuda.get_device_name(0),
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "conda_env": os.environ.get("CONDA_DEFAULT_ENV"), "taskset": run(["taskset", "-cp", str(os.getpid())]).strip(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
           "sparse_detr_commit": run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"]).strip(), "sparse_detr_dirty": bool(run(["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain"]).strip()),
           "script_sha256": sha256_file(Path(__file__).resolve()), "uptime": run(["uptime"]).strip(), "nvidia_smi_before": run(["nvidia-smi"]), "tf32": False, "cudnn_benchmark": False, "dtype": "float32", "smoke": a.smoke, "mode": a.mode,
           "pip_freeze": run([sys.executable, "-m", "pip", "freeze"]), "conda_list": run(["conda", "list", "-n", "sparse-detr"])}
    (a.output_dir / "env.json").write_text(json.dumps(env, indent=1) + "\n")
    model, info = build(device)
    spec = {"label": "sparse_detr_r50_rho0.4", "group": "sparse_detr", "config": "sparse-detr official rho=0.4 (R50)", "checkpoint": info["checkpoint"], "checkpoint_sha256": info["checkpoint_sha256"], "note": info["args"], "eval_size": [H, W]}
    (a.output_dir / "manifest_used.json").write_text(json.dumps({"configs": [spec], "set": "s4_sparse_detr_" + a.mode}, indent=1) + "\n")
    tl = GpuTimeline(); tl.start()
    results = []
    if a.mode == "reproduce":
        torch.manual_seed(0); x = torch.randn(1, 3, H, W, device=device)
        img_meta = {"input": "torch.randn(1,3,800,1333) seed 0 (the 2026-09 script used an unseeded randn; FLOPs do not depend on values)"}
        m = measure(model, x, a.warmups, a.samples, True)
        results.append({"label": spec["label"], "pass": 1, "spec": spec, "model": {"audit": info, "params": {"total": info["params_total"]}}, "images": img_meta, "t_start": env["time_start"], "measurements": [m], "t_end": now()})
        print("[reproduce] b=1 %.2f±%.2f ms %.6f GFLOPs (fvcore %.6f + supp %.6f) params %.6fM" % (m["latency_ms"]["mean"], m["latency_ms"]["sample_sd"], m["flops"]["accounted_total_gflops"], m["flops"]["fvcore_supported_gflops"], m["flops"]["msdeformattn_supplement_gflops"], info["params_total"] / 1e6), flush=True)
    else:
        x_all, img_meta = load_images(16, device)
        batches = [int(b) for b in a.batches.split(",")]
        for p in range(1, a.passes + 1):
            row = {"label": spec["label"], "pass": p, "spec": spec, "model": {"audit": info, "params": {"total": info["params_total"]}}, "images": img_meta, "t_start": now(), "measurements": []}
            order = list(batches); random.Random(0).shuffle(order)
            for b in order:
                m = measure(model, x_all[:b].contiguous(), a.warmups, a.samples, want_flops=(p == 1)); row["measurements"].append(m)
                print("[pass %d] b=%2d %s%s" % (p, b, ("%.2f±%.2f ms" % (m["latency_ms"]["mean"], m["latency_ms"]["sample_sd"])) if m["status"] == "ok" else m["status"], (" %.2f GFLOPs/img" % (m["flops"]["accounted_total_gflops"] / b)) if "flops" in m else ""), flush=True)
            row["measurements"].sort(key=lambda q: q["batch"]); row["t_end"] = now(); results.append(row)
    tl.stop.set(); tl.join(timeout=10)
    env["time_end"] = now(); env["nvidia_smi_after"] = run(["nvidia-smi"]); env["uptime_after"] = run(["uptime"]).strip()
    (a.output_dir / "env.json").write_text(json.dumps(env, indent=1) + "\n"); (a.output_dir / "timeline.json").write_text(json.dumps(tl.rows) + "\n")
    checks = []
    okr = [m for m in results[0]["measurements"] if m["status"] == "ok" and "flops" in m]
    if len(okr) > 1:
        per = [m["flops"]["accounted_total_gflops"] / m["batch"] for m in okr]; checks.append({"check": "gflops_batch_linear", "label": spec["label"], "per_image_gflops_by_batch": per, "passed": max(per) - min(per) < 1e-6 * max(per) + 1e-6})
    with (a.output_dir / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["label", "pass", "batch", "status", "gflops_per_image", "latency_mean_ms", "latency_sd_ms", "latency_median_ms", "images_per_s", "checkpoint_sha256"])
        for r in results:
            for m in r["measurements"]:
                fl = m.get("flops") or next((q.get("flops") for q in results[0]["measurements"] if q["batch"] == m["batch"]), None)
                w.writerow([r["label"], r["pass"], m["batch"], m["status"], "%.6f" % (fl["accounted_total_gflops"] / m["batch"]) if fl else "", "%.4f" % m["latency_ms"]["mean"] if m["status"] == "ok" else "", "%.4f" % m["latency_ms"]["sample_sd"] if m["status"] == "ok" else "", "%.4f" % m["latency_ms"]["median"] if m["status"] == "ok" else "", "%.3f" % m["throughput_images_per_s"] if m["status"] == "ok" else "", info["checkpoint_sha256"]])
    (a.output_dir / "results.json").write_text(json.dumps({"status": "complete", "smoke": a.smoke, "set": "s4_sparse_detr_" + a.mode, "protocol": {"dtype": "float32", "tf32": False, "cudnn_benchmark": False, "eval": True, "no_grad": True, "warmups": a.warmups, "samples": a.samples, "passes": a.passes, "batches": a.batches, "timing": "CUDA events, synchronize before and after every sample", "flops": "fvcore supported ops + analytic MSDeformAttn term"}, "structure_checks": checks, "results": results}, indent=1) + "\n")
    print("done ->", a.output_dir, flush=True)


if __name__ == "__main__":
    main()
