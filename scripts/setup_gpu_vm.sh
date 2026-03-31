#!/usr/bin/env bash
# ============================================================
# setup_gpu_vm.sh — Google Cloud L4 GPU setup
# Installer CUDA-drivere, Ollama med GPU-støtte og Python-miljø
# Kjør: bash scripts/setup_gpu_vm.sh
# ============================================================
set -euo pipefail

echo "=== [1/5] Oppdaterer systemet ==="
sudo apt-get update && sudo apt-get upgrade -y

echo "=== [2/5] Installerer NVIDIA CUDA-drivere ==="
# Sjekk om nvidia-smi allerede finnes
if command -v nvidia-smi &>/dev/null; then
    echo "NVIDIA-driver allerede installert:"
    nvidia-smi
else
    sudo apt-get install -y ubuntu-drivers-common
    sudo ubuntu-drivers autoinstall
    echo "Driver installert — reboot kan være nødvendig etter dette scriptet."
fi

echo "=== [3/5] Installerer Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

# Aktiver og start Ollama-tjenesten
sudo systemctl enable ollama
sudo systemctl start ollama

# Vent til Ollama er klar
echo "Venter på Ollama..."
for i in {1..10}; do
    if curl -s http://localhost:11434 &>/dev/null; then
        echo "Ollama er oppe."
        break
    fi
    sleep 2
done

echo "=== [4/5] Laster ned qwen2.5:32b ==="
ollama pull qwen2.5:32b

echo "=== [5/5] Verifiserer GPU-bruk ==="
echo ""
echo "--- nvidia-smi ---"
nvidia-smi

echo ""
echo "--- Ollama GPU-info ---"
ollama run qwen2.5:32b "Svar kun: GPU OK" --verbose 2>&1 | grep -E "gpu|GPU|vram|VRAM|load" || true

echo ""
echo "======================================"
echo "Setup ferdig!"
echo "Kjør på VMen:"
echo "  export \$(grep -v '^#' .env | xargs)"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "======================================"
