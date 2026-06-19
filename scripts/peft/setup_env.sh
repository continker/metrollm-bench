#!/bin/bash
# Setup PEFT training environment on the 5090 box
# Usage: HF_TOKEN=hf_xxx bash setup_env.sh
set -euo pipefail

PEFT_DIR="$HOME/metrollm-peft"
VENV="$PEFT_DIR/venv"
MODEL_DIR="$PEFT_DIR/Qwen3.5-2B"

echo "=== Creating PEFT environment at $PEFT_DIR ==="
mkdir -p "$PEFT_DIR"

# --- venv ---
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
    echo "Created venv."
else
    echo "venv exists, skipping."
fi

source "$VENV/bin/activate"

# --- torch (CUDA 12.8) ---
echo "=== Installing PyTorch 2.9 + CUDA 12.8 ==="
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "Torch check:"
python -c "import torch; print(f'  torch {torch.__version__}, cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# --- unsloth ---
echo "=== Installing unsloth ==="
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
# Core deps (unsloth may already pull most of these)
pip install trl peft accelerate bitsandbytes datasets transformers

echo "Unsloth check:"
python -c "import unsloth; print(f'  unsloth OK')"

# --- HuggingFace token ---
if [ -n "${HF_TOKEN:-}" ]; then
    echo "=== Logging in to HuggingFace ==="
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "WARNING: HF_TOKEN not set. Run: huggingface-cli login"
fi

# --- Download Qwen3.5-2B weights ---
if [ ! -d "$MODEL_DIR" ] || [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
    echo "=== Downloading Qwen/Qwen3.5-2B weights ==="
    huggingface-cli download Qwen/Qwen3.5-2B \
        --local-dir "$MODEL_DIR" \
        --exclude "*.gguf" "*.bin"
    echo "Download complete: $MODEL_DIR"
else
    echo "Model weights exist: $MODEL_DIR"
    ls "$MODEL_DIR"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $VENV/bin/activate"
echo "Model at:      $MODEL_DIR"
