# 🔊 SSRFNet Inference Demo (Colab)

This example demonstrates how to use the pretrained **SSRFNet** from [SSRFNet](https://github.com/dayflys/SSRFNet) to extract **speaker embeddings** from a single audio sample.

---

## 🚀 Open Colab
Go to [Google Colab](https://colab.research.google.com/drive/1DaVpnnwQd9Jg655RR5leDabZVNnjIo79?usp=sharing) and see few line example.

---

## ⚡ Quick Usage with Your Scripts
If you only want to use the pretrained models, you can easily load them with `torch.hub`:

```python
import torch

# Load model
front = torch.hub.load("dayflys/SSRFNet",'SSRFNet_svmixer', pretrained=True).eval()
backend = torch.hub.load("dayflys/SSRFNet",'SSRFNet_backend', pretrained=True).eval()

# Dummy waveform input (1 second at 48 kHz)
input_wav = torch.rand(1, 48000)

# Forward pass
with torch.no_grad():
    ptm_embedding = front(input_wav)
    embedding = backend(ptm_embedding,input_wav)

```
