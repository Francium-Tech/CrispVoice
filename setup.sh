#!/bin/sh
# One-shot setup: creates a fully self-contained environment inside this
# folder (Python interpreter, PyTorch, model weights, bundled ffmpeg).
# Nothing is installed system-wide; ./uninstall.sh removes everything.
#
# Requirements: uv (https://docs.astral.sh/uv/) and curl.
# Tested on macOS (Apple Silicon); Linux x86_64 should work (CPU wheels).
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

command -v uv >/dev/null 2>&1 || {
    echo "error: uv is required. Install it with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
}

echo "==> [1/5] Creating project-local Python 3.11 environment"
export UV_PYTHON_INSTALL_DIR="$DIR/.runtime"
uv venv --python 3.11 .venv

echo "==> [2/5] Installing PyTorch (CPU) and dependencies"
# torch 2.2.2: last line before the weights_only load change; numpy<2 required.
if [ "$(uname)" = "Linux" ]; then
    TORCH_INDEX="--index-url https://download.pytorch.org/whl/cpu"
else
    TORCH_INDEX=""
fi
uv pip install --python .venv/bin/python $TORCH_INDEX \
    "torch==2.2.2" "torchaudio==2.2.2" "torchvision==0.17.2"
uv pip install --python .venv/bin/python "numpy<2" \
    librosa omegaconf rich soundfile tqdm resampy tabulate pandas \
    matplotlib celluloid ptflops imageio-ffmpeg
# resemble-enhance declares deepspeed (training-only, does not build on
# macOS), so install without deps; everything inference needs is above.
uv pip install --python .venv/bin/python --no-deps resemble-enhance==0.0.1

echo "==> [3/5] Creating deepspeed stub (training-only dep, unused at inference)"
.venv/bin/python - <<'EOF'
import sysconfig, pathlib
sp = pathlib.Path(sysconfig.get_paths()["purelib"])
root = sp / "deepspeed"
(root / "runtime").mkdir(parents=True, exist_ok=True)
msg = 'raise RuntimeError("deepspeed stub: training is not supported in this install")'
(root / "__init__.py").write_text(
    "# Stub for inference-only use of resemble-enhance.\n"
    f"__version__ = '0.0.0-stub'\n\n"
    f"class DeepSpeedConfig:\n    def __init__(self, *a, **k):\n        {msg}\n\n"
    f"def init_distributed(*a, **k):\n    {msg}\n")
(root / "accelerator.py").write_text(f"def get_accelerator():\n    {msg}\n")
(root / "runtime" / "__init__.py").write_text("")
(root / "runtime" / "engine.py").write_text(
    f"class DeepSpeedEngine:\n    def __init__(self, *a, **k):\n        {msg}\n")
(root / "runtime" / "utils.py").write_text(f"def clip_grad_norm_(*a, **k):\n    {msg}\n")
print(f"stub written to {root}")
EOF

echo "==> [3b/5] Creating DeepFilterNet environment (blend stage; needs older torch)"
uv venv --python 3.11 .venv-dfn
uv pip install --python .venv-dfn/bin/python deepfilternet \
    "torch==2.1.2" "torchaudio==2.1.2" "numpy<2" soundfile

echo "==> [3c/5] Creating ClearerVoice environment (clarity front-end)"
uv venv --python 3.11 .venv-cv
uv pip install --python .venv-cv/bin/python clearvoice
# its MossFormer2_SE_48K checkpoint (~211 MB) downloads into models/clearvoice
# on first run

echo "==> [3d/5] Creating GPU fast-path environment (macOS Apple Silicon)"
if [ "$(uname)" = "Darwin" ]; then
    uv venv --python 3.11 .venv-gpu
    uv pip install --python .venv-gpu/bin/python "torch==2.7.1" "torchaudio==2.7.1" \
        "numpy<2" soundfile
    uv pip install --python .venv-gpu/bin/python --no-deps resemble-enhance==0.0.1
    uv pip install --python .venv-gpu/bin/python librosa omegaconf rich tqdm resampy \
        tabulate pandas matplotlib celluloid ptflops
    GPU_SP=$(.venv-gpu/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
    cp -R "$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/deepspeed" "$GPU_SP/" 2>/dev/null || true
else
    echo "    (skipped: not macOS; the pipeline runs CPU-only elsewhere)"
fi

echo "==> [4/5] Downloading Resemble Enhance weights (~713 MB, one time)"
HF="https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2"
mkdir -p models/enhancer_stage2/ds/G/default
[ -f models/enhancer_stage2/hparams.yaml ] || \
    curl -L -o models/enhancer_stage2/hparams.yaml "$HF/hparams.yaml"
[ -f models/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt ] || \
    curl -L -o models/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt \
        "$HF/ds/G/default/mp_rank_00_model_states.pt"

echo "==> [5/5] Verifying installation"
.venv/bin/python -c "
from resemble_enhance.enhancer.inference import enhance
import torch
print('ok: torch', torch.__version__)
"
chmod +x enhance uninstall.sh

echo
echo "Done. Try it:  ./enhance your_recording.mp3"
