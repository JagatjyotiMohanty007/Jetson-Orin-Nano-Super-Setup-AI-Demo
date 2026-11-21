#!/usr/bin/env bash
set -u
printf '\n=== Device ===\n'
if [[ -r /proc/device-tree/model ]]; then tr -d '\0' </proc/device-tree/model; echo; else echo 'Device tree model unavailable'; fi
printf '\n=== OS ===\n'
cat /etc/os-release 2>/dev/null | grep -E '^(PRETTY_NAME|VERSION)=' || true
printf '\n=== Architecture ===\n'
uname -m
printf '\n=== Jetson Linux / L4T ===\n'
cat /etc/nv_tegra_release 2>/dev/null || echo 'Not found'
printf '\n=== JetPack package ===\n'
dpkg-query -W nvidia-jetpack 2>/dev/null || echo 'Not installed or unavailable'
printf '\n=== CUDA compiler ===\n'
if command -v nvcc >/dev/null 2>&1; then nvcc --version; elif [[ -x /usr/local/cuda/bin/nvcc ]]; then /usr/local/cuda/bin/nvcc --version; else echo 'nvcc not found'; fi
printf '\n=== Python and PyTorch ===\n'
python3 --version
python3 - <<'PY'
try:
    import torch
    print('PyTorch:', torch.__version__)
    print('PyTorch CUDA build:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
except ImportError:
    print('PyTorch not installed in current Python environment')
PY
printf '\n=== RAM and disk ===\n'
free -h
df -h / | tail -n 1
