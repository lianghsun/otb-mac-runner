# OpenTWBench · Apple Silicon (MLX) runner

Scores models on OpenTWBench that the Linux 3090 box **can't serve** — its
NVIDIA driver caps at CUDA 12.5 and a GeForce card has no forward-compat, so a
vLLM new enough for **Qwen3.5 / LFM2 / Granite-4.0 (mamba hybrid)** won't start
there. Apple Silicon runs them on Metal, and unified memory fits models the
24 GB card can't.

**Privacy:** the benchmark items are pulled from a *private* HF repo with your
token and never written here; results are only a per-question hash plus
booleans (correct? box-parsed? lenient-parsed?). So this repo is public but the
benchmark isn't in it.

## Run

```sh
git clone https://github.com/lianghsun/otb-mac-runner.git
cd otb-mac-runner
HF_TOKEN=hf_xxx bash bootstrap.sh          # installs MLX, runs everything
```

Formosa (2,665 q) runs first for every model, then the exam sample (22,200 q).
Resumable — rerun and it skips finished questions. Results auto-commit + push
to the `mac-results` branch every 5 min, where the Linux box folds them onto
the leaderboard.

```sh
MODELS='LiquidAI/LFM2-1.2B' HF_TOKEN=hf_xxx bash bootstrap.sh   # just one
```

Big models: `sudo sysctl iogpu.wired_limit_mb=229376` (bootstrap tries it).

## Models

Edit `DEFAULT_MODELS` in `eval_mlx.py`. Defaults: Ornith-1.5-9B (Qwen3.5),
LFM2-1.2B/700M, granite-4.0 h-small/h-tiny/micro. Anything your `mlx-lm` /
`mlx-vlm` can't load is listed FAILED in `results/_summary.json` — upgrade
(`pip install -U mlx-lm mlx-vlm`) and rerun.
