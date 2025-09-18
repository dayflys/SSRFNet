# 🧪 In-Depth Evaluation (Evaluation-Only Mode)

This directory provides scripts to **evaluate pretrained SSRFNet model** on multiple benchmark datasets without training.

## 📊 Supported Benchmarks

Running the evaluation will automatically test on:

- **VoxCeleb1-O**
- **VoxCeleb1-E**
- **VoxCeleb1-H**
- **VOiCES**
- **VoxSRC2023**
- **VCMix**


## ▶ How to Run

1. **Download**

   Download the `./test_code` scripts from this repository and make sure they are available in your local environment.
   These scripts contain the evaluation pipeline for all supported benchmarks.

2. **Edit arguments**  
   Open [`test_code/arguments.py`](./test_code/arguments.py) and update the following fields:

   - `trial_file`: path to the downloaded evaluation trial file (see [📂 Trial Files](https://github.com/Jungwoo4021/experimental-resources/tree/main/test_trials))
     - vox2_testO_trials.txt  
     - vox2_testE_trials.txt  
     - vox2_testH_trials.txt  
     - vcmix_test.txt  
     - voxsrc_test.txt  
     - voices_eval.txt  

3. **Run the script**

   ```bash
   python test_code/main.py

## Model Performance

- **SSRFNet**    
  - Size: 75.1M parameters  
  - Vox1-O EER: 0.601%
  - Vox1-E EER: 0.800%
  - Vox1-H EER: 1.446%
  - VOiCES EER: 8.55%
  - VoxSRC23 EER: 4.14%
  - VCMix EER: 2.47%
