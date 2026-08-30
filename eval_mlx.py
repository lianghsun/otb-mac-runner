#!/usr/bin/env python3
"""OpenTWBench eval on Apple Silicon via MLX — for the models the Linux 3090
box can't serve (driver caps at CUDA 12.5; GeForce has no forward-compat, so a
vLLM new enough for Qwen3.5 / LFM2 / Granite-4.0 mamba won't run there).

Privacy by design: the benchmark items are pulled from a PRIVATE Hugging Face
repo with your HF_TOKEN and never written to disk in the clear; results contain
only a hash of each question plus booleans (correct? box-parsed? lenient-
parsed?), so they can live in a public repo without leaking the benchmark.

Protocol matches the main suite exactly — box extraction, deterministic
question-hash option shuffle, temperature 0, the same system prompt — and the
box-first + bare-letter lenient fallback is applied here on the Mac.

    HF_TOKEN=hf_xxx python eval_mlx.py --benches formosa
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import time

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "results"
DATA_REPO = "OpenTWBench/otb-mac-data"          # private
BENCHES = {"formosa": "formosa.parquet", "exam": "exam-sample.parquet"}

SYSTEM = (
    "你是一位專業的測驗作答助理。請仔細閱讀題目，以繁體中文（臺灣用語）思考，"
    "並在最後輸出 \\boxed{}，大括號內只填入唯一正確選項的英文字母，也就是 "
    "A、B、C、D 其中一個。除了 \\boxed{} 之外不要使用其他格式標示答案。")

_BOX = re.compile(r"\\boxed\{\s*([A-Da-d甲乙丙丁])\s*\}")
_ORD = {"甲": "A", "乙": "B", "丙": "C", "丁": "D"}
_LEAD = re.compile(r"^\s*[（(]?\s*([A-Da-d甲乙丙丁])\s*[)）.。、:：\s]")
_MARKED = re.compile(r"(?:答案|正確選項|正解|答|選)\s*(?:是|為|:|：)?\s*[（(]?\s*([A-Da-d甲乙丙丁])")
_ANY = re.compile(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])")

DEFAULT_MODELS = [
    # --- driver-blocked small models (the reason we need the Mac at all) ---
    "ornith-ai/Ornith-1.5-9B",
    "LiquidAI/LFM2-1.2B",
    "LiquidAI/LFM2-700M",
    "ibm-granite/granite-4.0-h-small",
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-micro",
    # --- big models the 24GB 3090 can't fit; full precision (bf16), one at a
    #     time, weights reclaimed after each so peak disk stays low. Ordered
    #     small -> large so the faster ones produce scores first. (g) = gated:
    #     accept the license once on the HF model page with this token's
    #     account, or it FAILs with 403 (logged, not fatal). ~GB = bf16 weights.
    "tencent/Hunyuan-7B-Instruct",               # ~14GB   (g?, arch may be unsupported)
    "THUDM/glm-4-9b-chat",                       # ~18GB
    "01-ai/Yi-1.5-9B-Chat",                      # ~18GB
    "mistralai/Mistral-Nemo-Instruct-2407",      # ~24GB   (g)
    "microsoft/Phi-4",                            # ~28GB   14B dense
    "Qwen/Qwen3-30B-A3B",                         # MoE, cheap to run
    "deepseek-ai/DeepSeek-V2-Lite-Chat",          # ~31GB MoE, non-reasoning
    "mistralai/Mistral-Small-24B-Instruct-2501", # ~48GB   (g)
    "google/gemma-3-27b-it",                      # ~54GB   (g)
    "Qwen/Qwen2.5-32B-Instruct",                  # ~64GB
    "zai-org/GLM-4-32B-0414",                     # ~64GB
    "Qwen/Qwen3-32B",                             # ~64GB
    "01-ai/Yi-1.5-34B-Chat",                      # ~68GB
    "mistralai/Mixtral-8x7B-Instruct-v0.1",       # ~93GB MoE  (g)
    "Qwen/Qwen2.5-72B-Instruct",                  # ~145GB flagship upper anchor
    "tencent/Hunyuan-A13B-Instruct",              # ~160GB MoE  (g?, arch may be unsupported)
    # --- reasoning models LAST: they think for thousands of tokens before the
    #     \boxed{}, so formosa is slow and the 22,200-q exam can take weeks each.
    #     formosa runs first per model and auto-pushes, so a partial run still
    #     scores them. They get max_tokens=16384 (see MAX_TOKENS). ---
    "Qwen/QwQ-32B",                               # ~64GB reasoning
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",   # ~28GB reasoning
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",   # ~64GB reasoning
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",  # ~140GB reasoning
    # Can't fit at bf16 on 256GB — quantize-only, off the board by rule:
    #   Kimi-K2 (~2TB), DeepSeek-R1 / V3 (~1.3TB), MiniMax-Text-01 (~912GB),
    #   Hunyuan-Large (~780GB), GLM-4.5 (~710GB),
    #   Mistral-Large-2411 (~246GB > 224GB wired).
]

# Reasoning models need room to finish thinking before the \boxed{} answer;
# 4096 truncates them mid-thought. Everything else stays at the CLI default.
MAX_TOKENS = {
    "Qwen/QwQ-32B": 16384,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 16384,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": 16384,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": 16384,
}


def qhash(stem: str) -> str:
    return hashlib.sha256(stem.encode()).hexdigest()[:16]


def box(text: str):
    m = _BOX.search(text or "")
    if not m:
        return None
    g = m.group(1).upper()
    return _ORD.get(g, g)


def lenient(text: str):
    if not text:
        return None
    t = text.split("</think>")[-1]
    for rx in (_BOX, _LEAD, _MARKED):
        m = rx.search(t)
        if m:
            g = m.group(1).upper()
            return _ORD.get(g, g)
    m = _ANY.search(t)
    return m.group(1).upper() if m else None


def purge_model(model):
    """Delete this model's downloaded weights from the HF cache after scoring,
    so the borrowed Mac's disk never holds more than one model at a time. Only
    the model snapshot is removed — the private benchmark parquet stays cached."""
    import shutil
    cache = (os.environ.get("HF_HUB_CACHE")
             or (os.environ.get("HF_HOME") and pathlib.Path(os.environ["HF_HOME"]) / "hub")
             or pathlib.Path.home() / ".cache" / "huggingface" / "hub")
    d = pathlib.Path(cache) / ("models--" + model.replace("/", "--"))
    if d.exists():
        try:
            freed = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        except Exception:
            freed = 0
        shutil.rmtree(d, ignore_errors=True)
        print(f"    reclaimed weights: {model}  (~{freed / 1e9:.1f} GB)", flush=True)


def load_rows(parquet):
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    p = hf_hub_download(DATA_REPO, parquet, repo_type="dataset",
                        token=os.environ.get("HF_TOKEN"))
    return pq.read_table(p).to_pylist()


def shuffled(row):
    opts = [(k, row[k]) for k in "ABCD"]
    seed = int.from_bytes(hashlib.sha256(row["question"].encode()).digest()[:8], "big")
    random.Random(seed).shuffle(opts)
    correct_text = row[row["answer"]]
    out = {"question": row["question"]}
    ans = None
    for (ok, text), nk in zip(opts, "ABCD"):
        out[nk] = text
        if text == correct_text:
            ans = nk
    out["answer"] = ans
    return out


def run_model(model, benches, max_tokens):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    print(f"\n=== loading {model}", flush=True)
    t0 = time.time()
    mdl, tok = load(model)
    print(f"    loaded in {time.time() - t0:.0f}s", flush=True)
    sampler = make_sampler(temp=0.0)
    out_dir = OUT / model.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    for bench in benches:
        rows = load_rows(BENCHES[bench])
        path = out_dir / f"{bench}.jsonl"
        done = set()
        if path.exists():
            for line in path.open():
                try:
                    done.add(json.loads(line)["qh"])
                except Exception:
                    pass
        print(f"    {bench}: {len(rows)} items, {len(done)} already done", flush=True)
        t0 = time.time()
        with path.open("a", encoding="utf-8") as fh:
            for i, row in enumerate(rows):
                qh = qhash(row["question"])
                if qh in done:
                    continue
                q = shuffled(row)
                body = (f"題目：{q['question']}\nA. {q['A']}\nB. {q['B']}\n"
                        f"C. {q['C']}\nD. {q['D']}")
                prompt = tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": body}],
                    add_generation_prompt=True, tokenize=False)
                text = generate(mdl, tok, prompt, max_tokens=max_tokens,
                                sampler=sampler, verbose=False)
                b = box(text)
                le = lenient(text)
                fh.write(json.dumps({
                    "qh": qh,
                    "ok": le == q["answer"],       # lenient-scored correctness
                    "boxp": b is not None,          # did it emit \boxed{}
                    "lenp": le is not None,         # did lenient read an answer
                }) + "\n")
                if (i + 1) % 100 == 0:
                    fh.flush()
                    rate = (i + 1 - len(done)) / max(1e-9, time.time() - t0)
                    print(f"      {i + 1}/{len(rows)}  {rate:.1f} q/s", flush=True)
        print(f"    {bench} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--benches", nargs="*", default=list(BENCHES))
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--keep", action="store_true",
                    help="keep weights after each model (default: reclaim)")
    args = ap.parse_args()

    counts: dict[str, int] = {}

    def complete(model):
        """True if every bench already has all its rows scored — checked from
        the JSONL line counts alone, so a finished model is skipped without
        re-downloading its weights (the current run scores under older code that
        never wrote a .done marker)."""
        d = OUT / model.replace("/", "_")
        for b in args.benches:
            counts.setdefault(b, len(load_rows(BENCHES[b])))
            p = d / f"{b}.jsonl"
            n = sum(1 for _ in p.open()) if p.exists() else 0
            if n < counts[b]:
                return False
        return True

    ok, failed, skipped = [], [], []
    for model in args.models:
        marker = OUT / model.replace("/", "_") / ".done"
        if marker.exists() or complete(model):   # scored already — don't re-download
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("done\n")
            print(f"\n=== skip {model} (already done)", flush=True)
            skipped.append(model)
            continue
        try:
            run_model(model, args.benches, MAX_TOKENS.get(model, args.max_tokens))
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("done\n")     # so a rerun won't re-download it
            ok.append(model)
        except Exception as e:
            print(f"    !! {model} failed: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            failed.append((model, f"{type(e).__name__}: {str(e)[:200]}"))
        finally:
            if not args.keep:               # reclaim disk whether it passed or failed
                purge_model(model)

    print("\n===== SUMMARY =====")
    for m in ok:
        print(f"  OK      {m}")
    for m in skipped:
        print(f"  SKIP    {m}  (already done)")
    for m, why in failed:
        print(f"  FAILED  {m}  ({why})")
    (OUT / "_summary.json").write_text(json.dumps(
        {"ok": ok, "skipped": skipped, "failed": failed},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
