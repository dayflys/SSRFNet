# 🔊 SSRFNet Inference Demo (Colab)

This example demonstrates how to use the pretrained **SSRFNet** from [SSRFNet](https://github.com/dayflys/SSRFNet) to extract **speaker embeddings** from a single audio sample.

---

## 🚀 Open Colab
<!-- Go to [Google Colab]() and see the 5-line example. -->

---

## ⚡ Quick Usage with Your Scripts
If you only want to use the pretrained models, you can easily load them with `torch.hub`:

```python
import torch

ssrfnet = torch.hub.load("dayflys/ssrfnet", "ssrfnet", pretrained=True).eval()

with torch.no_grad():
    embedding = ssrfnet(input_wav)
```
