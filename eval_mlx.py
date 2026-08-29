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
    "ornith-ai/Ornith-1.5-9B",
    "LiquidAI/LFM2-1.2B",
    "LiquidAI/LFM2-700M",
    "ibm-granite/granite-4.0-h-small",
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-micro",
]


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
    args = ap.parse_args()

    ok, failed = [], []
    for model in args.models:
        try:
            run_model(model, args.benches, args.max_tokens)
            ok.append(model)
        except Exception as e:
            print(f"    !! {model} failed: {type(e).__name__}: {str(e)[:200]}",
                  flush=True)
            failed.append((model, f"{type(e).__name__}: {str(e)[:200]}"))

    print("\n===== SUMMARY =====")
    for m in ok:
        print(f"  OK      {m}")
    for m, why in failed:
        print(f"  FAILED  {m}  ({why})")
    (OUT / "_summary.json").write_text(json.dumps(
        {"ok": ok, "failed": failed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
