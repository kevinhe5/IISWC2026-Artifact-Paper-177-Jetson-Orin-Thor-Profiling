#!/usr/bin/env python3
"""Assemble kernel_categories_thor_gen65536.json (Fig8 long-context pane) from the
per-fw dicts appended by longctx_capture.sh to logs/longctx_g65536.jsonl.
Adds provenance + a bundled 5-cat summary. Frameworks that failed to capture
(e.g. sglang spawned-worker) are simply absent and listed under _missing."""
import json, os
from pathlib import Path

_WORK = os.environ.get("PROFILE_ROOT", "/nvme/iiswc/Jetson_profile") + "/work"
JL = Path(os.environ.get("THOR_LONGCTX_JSONL", f"{_WORK}/logs/longctx_g65536.jsonl"))
OUT = Path(__file__).resolve().parents[4] / "data/nsys/kernel_categories_thor_gen65536.json"
GEN = 65536
EXPECT = ["trtedge_llm", "llamacpp", "vllm", "sglang", "pytorch"]
NOTES = {
    "llamacpp": "llama.cpp f16 GGUF, flash_attn=1; grid-aware attention split (flash/soft_max named + bmm gridY split)",
    "pytorch": "PyTorch eager bf16; attention via bmm -> grid-aware split (gridY=heads, gridX<=256)",
    "vllm": "vLLM 0.12 fp16 enforce_eager; unified_attention named + grid split",
    "sglang": "SGLang flashinfer -- spawned scheduler escapes nsys fork-trace; long-ctx torch-profiler infeasible at 65536 steps (see _missing)",
    "trtedge_llm": "TRT-Edge-LLM fp16; name-only (Myelin/cutlass=matmul, gemv_mha/fmha=attention)",
}


def bundle(cats):
    b = {k: 0.0 for k in ("matmul", "attention", "quantize", "copy_cast", "other")}
    for n, v in cats.items():
        b[n if n in b else "other"] += v["ms_per_tok"]
    return {k: round(v, 4) for k, v in b.items()}


def main():
    merged = {}
    if JL.exists():
        for line in JL.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                merged.update(json.loads(line))
            except Exception:
                pass
    out = {}
    for fw, r in merged.items():
        if isinstance(r, dict) and "categories" in r:
            r["_note"] = NOTES.get(fw, "")
            r["_bundle5"] = bundle(r["categories"])
            out[fw] = r
    out["_platform"] = "thor"
    out["_gen_tokens"] = GEN
    out["_pp"] = 128
    out["_capture"] = "decode-only nsys (--capture-range=cudaProfilerApi), pp=128, gen=65536, Llama-3.2-1B; grid-aware attention split (attention QK/AV via bmm grows with KV length)"
    out["_missing"] = [fw for fw in EXPECT if fw not in out]
    OUT.write_text(json.dumps(out, indent=2))
    print("wrote", OUT)
    for fw in EXPECT:
        if fw in out:
            b = out[fw]["_bundle5"]
            print(f"  {fw:12s} total={out[fw]['total_ms']/GEN:.2f} matmul={b['matmul']:.2f} attn={b['attention']:.2f} other={b['other']:.2f}")
        else:
            print(f"  {fw:12s} MISSING")


if __name__ == "__main__":
    main()
