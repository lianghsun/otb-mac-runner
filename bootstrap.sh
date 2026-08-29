#!/bin/bash
# OpenTWBench — Apple Silicon (MLX) runner for the models the Linux 3090 can't
# serve (Qwen3.5 / LFM2 / Granite-4.0). Public repo; the benchmark itself stays
# private — pulled from a private HF repo with your token, and results are just
# per-question hashes + booleans, so nothing sensitive lives here.
#
#   git clone https://github.com/lianghsun/otb-mac-runner.git
#   cd otb-mac-runner
#   HF_TOKEN=hf_xxx bash bootstrap.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! set HF_TOKEN first (reads the private benchmark + gated models):"
  echo "   HF_TOKEN=hf_xxx bash bootstrap.sh"
  exit 1
fi
export HF_TOKEN

echo "== 1. Python venv + MLX"
PYBIN="$(command -v python3.11 || command -v python3)"
"$PYBIN" -m venv .venv
./.venv/bin/pip -q install --upgrade pip
./.venv/bin/pip -q install "mlx-lm>=0.28" "mlx-vlm>=0.1" pyarrow "huggingface_hub" || \
  ./.venv/bin/pip -q install "mlx-lm" "mlx-vlm" pyarrow "huggingface_hub"
./.venv/bin/python -c "import mlx.core as mx; print('   MLX ok, device:', mx.default_device())"

echo "== 2. lift the GPU wired-memory limit (unified memory — let MLX use it all)"
sudo sysctl iogpu.wired_limit_mb=229376 2>/dev/null || \
  echo "   (skipped — run 'sudo sysctl iogpu.wired_limit_mb=229376' by hand for big models)"

echo "== 3. background auto-push: commit+push new results every 5 min"
push_now(){ git add results 2>/dev/null || true; \
  git -c user.email=lianghsunh@gmail.com -c user.name="Liang-Hsun Huang" \
    commit -q -m "results $(date +%H:%M)" 2>/dev/null && \
    git push -q origin HEAD:mac-results 2>/dev/null && echo "   [auto-push $(date +%H:%M)]" || true; }
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  # self-update: pull the latest code from main, then branch results off it
  git checkout main >/dev/null 2>&1 && git pull -q origin main 2>/dev/null || true
  git checkout -B mac-results >/dev/null 2>&1 || true
  ( while true; do sleep 300; push_now; done ) &
  PUSHER=$!
  trap 'kill $PUSHER 2>/dev/null || true; push_now' EXIT
  echo "   auto-push armed (pid $PUSHER) — results stream to branch mac-results"
fi

echo "== 4. run the eval (per model: formosa then exam, then reclaim its weights)"
# One pass, one model at a time: score formosa+exam, then delete that model's
# weights before downloading the next — so the borrowed Mac never holds more
# than one model on disk. A rerun skips models already marked .done.
# (macOS ships bash 3.2, which errors on an empty array under `set -u`; branch
# on MODELS instead of building an array.)
if [ -n "${MODELS:-}" ]; then
  ./.venv/bin/python eval_mlx.py --models ${MODELS}
else
  ./.venv/bin/python eval_mlx.py
fi

echo "== 5. done. Results auto-pushed to the mac-results branch throughout."
echo "     (manual: git add results && git commit -m results && git push origin HEAD:mac-results)"
