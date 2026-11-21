# NVIDIA Jetson Orin Nano Super — Setup and First AI Demo

A reproducible, beginner-friendly setup guide and **PyTorch neural-network training demo** for the NVIDIA Jetson Orin Nano Super Developer Kit. The code also works on other Linux computers with compatible PyTorch builds.

> **Important:** The Jetson uses **ARM64 (aarch64)** and an NVIDIA JetPack-specific CUDA stack. Do **not** blindly install an ordinary PyPI `torch` wheel and assume it will use the Jetson GPU. Choose a PyTorch build/container compatible with your **exact JetPack / Jetson Linux release**.

## 1. Hardware checklist

- NVIDIA Jetson Orin Nano Super Developer Kit (8 GB)
- Appropriate power supply
- microSD card with a supported JetPack image, or correctly configured NVMe boot storage
- DisplayPort monitor (or compatible DisplayPort-to-HDMI adapter), USB keyboard and mouse
- Ethernet or supported Wi-Fi connection
- Optional NVMe SSD for datasets, environments and models

The board is a **GPU-based classical computer**, not a quantum processing unit.

## 2. First boot / operating-system installation

1. Check the [official Jetson Getting Started guide](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit) and the [JetPack SDK page](https://developer.nvidia.com/embedded/jetpack) for the image and flashing method supported by your board and boot firmware.
2. Flash the supported JetPack image to the designated storage using the official instructions. **A generic desktop Ubuntu ISO is not a substitute for a Jetson-compatible image.**
3. Connect display, keyboard, mouse, networking and power.
4. Complete the first-boot Ubuntu account and password setup.
5. Open a terminal and run the environment checks below.

**Note:** Bootloader firmware, flashing instructions, supported Ubuntu releases and JetPack versions can change. Follow NVIDIA's guide for your installed release rather than assuming a particular JetPack version.

## 3. Inspect the system

```bash
cat /proc/device-tree/model
uname -m
cat /etc/os-release
cat /etc/nv_tegra_release
apt list --installed 2>/dev/null | grep nvidia-jetpack
nvcc --version
free -h
lsblk
```

Run the included report:

```bash
bash scripts/check_environment.sh
```

Expected architecture on Jetson: `aarch64`.

## 4. Basic developer tools

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-dev \
    git curl wget build-essential cmake
```

If JetPack components are missing, consult NVIDIA's release-specific installation instructions before running `sudo apt install nvidia-jetpack`. That package must match your configured NVIDIA repositories and Jetson Linux release.

## 5. Set up Python and GPU-enabled PyTorch

**First determine the exact versions** using `bash scripts/check_environment.sh`.

### Option A: Already have working PyTorch

If `python3 -c "import torch; print(torch.cuda.is_available())"` prints `True`, use that Python environment to run the demo. Do not reinstall PyTorch unnecessarily.

### Option B: Need PyTorch

Use NVIDIA's Jetson-compatible PyTorch instructions, wheel or container for your **specific JetPack release**:

- [NVIDIA PyTorch for Jetson documentation](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [NVIDIA Jetson containers](https://github.com/dusty-nv/jetson-containers) (community-maintained; check release compatibility)

A virtual environment can isolate project packages:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip --version
```

`--system-site-packages` allows access to an existing system-installed PyTorch; it does **not** install PyTorch. If using a NVIDIA-provided wheel, follow its dependency and installation instructions within a compatible environment. Do not add `torch` to a generic `pip install -r requirements.txt` step.

Verify:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

## 6. Run the first AI Python demo

The demo generates synthetic two-dimensional samples, learns a **nonlinear binary classifier** with a small neural network, evaluates a held-out test set, and reports training time.

```bash
python src/ai_demo.py
```

Optional parameters:

```bash
python src/ai_demo.py --epochs 150 --samples 4000 --seed 42
```

**Example output (illustrative, not a measured Jetson benchmark):**

```text
PyTorch: <installed version>
PyTorch CUDA build: <installed CUDA version>
CUDA available: True
Compute device: cuda
GPU: Orin
Epoch   1/100 | loss: ...
...
Test accuracy: ...
Training time: ... s
Demo completed successfully.
```

If CUDA is unavailable, the script **falls back to CPU** and prints a warning. This is useful for testing the Python code but **does not demonstrate GPU acceleration**.

## 7. Monitor Jetson resource use

In a second terminal, run:

```bash
tegrastats
```

You can also inspect power modes with `sudo nvpmodel -q`. Do not change power modes or lock clocks without checking thermal limits, cooling and the installed power profile.

## 8. Troubleshooting

| Symptom | Check |
|---|---|
| `ModuleNotFoundError: torch` | Activate the environment containing the Jetson-compatible PyTorch build. |
| `CUDA available: False` | Confirm JetPack version, NVIDIA drivers/libraries, and CUDA-enabled PyTorch build. |
| `nvcc: command not found` | Check `/usr/local/cuda/bin/nvcc` and JetPack installation; PyTorch may work even without `nvcc` in PATH. |
| `Killed` or out-of-memory | Reduce `--samples`; close other memory-intensive processes. |
| `No space left on device` | Check `df -h` and consider NVMe storage. |
| Slow training | This is a deliberately tiny model; GPU launch overhead may dominate. It is a functional demo, not a speed benchmark. |

## 9. Repository layout

```text
jetson-orin-nano-super-ai-demo/
├── README.md
├── src/
│   └── ai_demo.py
├── scripts/
│   └── check_environment.sh
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 10. Push to GitHub

Create an **empty** repository on GitHub, then run from this project directory:

```bash
git init
git add .
git commit -m "Document Jetson Orin Nano Super setup and add first PyTorch AI demo"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/jetson-orin-nano-super-ai-demo.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Authenticate using your configured Git credential manager, SSH, or supported personal access token flow; GitHub account passwords are not accepted for HTTPS Git operations.

## 11. References

- [NVIDIA Jetson Orin Nano getting started](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit)
- [NVIDIA JetPack](https://developer.nvidia.com/embedded/jetpack)
- [NVIDIA PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)

## License

MIT — see [LICENSE](LICENSE).
