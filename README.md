
[![Torch Hub](https://img.shields.io/badge/Supported-Torch%20Hub-orange?logo=pytorch)](https://colab.research.google.com/drive/1DaVpnnwQd9Jg655RR5leDabZVNnjIo79?usp=sharing) 

# SSRFNet : Stage-wise SV-Mixer and RedimNet Fusion Network for Speaker Verification

> [📄 Paper: *SSRFNet : Stage-wise SV-Mixer and RedimNet Fusion Network for Speaker Verification*](./assets/paper.pdf)  
> 📅 Conference: IEEE ICASSP 2026 (submitted)  
> 🚀 **Now available on [PyTorch Hub](https://colab.research.google.com/drive/1DaVpnnwQd9Jg655RR5leDabZVNnjIo79?usp=sharing)!**  
> (Load pretrained **SSRFNet** models in just few line of code)

---

## 🚀 Usage via Torch Hub

```python
front = torch.hub.load("dayflys/SSRFNet",'SSRFNet_svmixer', pretrained=True).eval()
backend = torch.hub.load("dayflys/SSRFNet",'SSRFNet_backend', pretrained=True).eval()

ptm_embedding = front(input_wav)
embedding = backend(ptm_embedding, input_wav)
```

## 🔍 Overview


---

## 🧠 Key Contributions

---

## 🏗️ Architecture

The SSRFNet pipeline consists of:

1. **SV-Mixer Encoder Blocks** (12 layers, 1024 hidden dimension)
2. **SSRFNet** (based on [ReDimNet](https://github.com/IDRnD/redimnet))

![SSRFNet Architecture](./assets/images/SSRFNet_architecture.png)

---

## 📊 Experimental Results


---

## ⚙️ Setup (Optional)

### 🐳 Docker (Recommended)

This project is designed to run inside a Docker container.

- The [`Dockerfile`](./assets/setup/Dockerfile) defines the full environment with CUDA and PyTorch support.
- Use the provided shell scripts ([`build.sh`](./assets/setup/build.sh), [`launch.sh`](./assets/setup/launch.sh)) to build and launch the container:

⚠️ **Important:** You **must** edit `{PATH_Dockerfile}` in [`build.sh`](./assets/setup/build.sh) before running.

```bash
# Build Docker image
./build.sh

# Launch container
./launch.sh
```

### 🛠 Manual Installation

For users who prefer **not to use Docker**, two options are provided:

- [`requirements.txt`](./assets/setup/requirements.txt): full environment with all dependencies (recommended for development and training)
- [`cleaned_requirements.txt`](./assets/setup/cleaned_requirements.txt): minimal environment for inference or lightweight usage

Install either one using:

```bash
# Full environment
pip install -r assets/setup/requirements.txt

# OR: Minimal setup
pip install -r assets/setup/cleaned_requirements.txt
```

## 🔧 Usage

This project supports three typical use cases:

### Option 1: Train & Evaluate (Full Pipeline)

Train a SSRFNet from scratch using WavLM knowledge distillation, and evaluate it on VoxCeleb1-O datasets. 

➝ Go to [`📁experiments/train`](./experiments/train/README.md)

### Option 2: In-Depth Evaluation (No Training)

Run detailed evaluations on multiple datasets (VoxCeleb-Hard, VC-Mix, VOiCES, ...) using a **pretrained model**.

➝ Go to [`📁experiments/eval_only`](./experiments/eval_only/README.md)

### Option 3: Inference-Only Mode (Using Only Speaker Embeddings) colab↗

Use this option if you only need a **pretrained speaker verification model** for quick testing or downstream tasks — no training or evaluation setup required. You can directly load a pretrained SSRFNet model with few line:

```python
front = torch.hub.load("dayflys/SSRFNet",'SSRFNet_svmixer', pretrained=True).eval()
backend = torch.hub.load("dayflys/SSRFNet",'SSRFNet_backend', pretrained=True).eval()
```

➝ Go to [`📁experiments/inference`](./experiments/inference/README.md)

## 📎 Citation

```bash
The citation will be added after the paper is accepted to ICASSP 2026.
```

## 🛡️ License

This project is licensed under the MIT License – see the LICENSE file for details.
