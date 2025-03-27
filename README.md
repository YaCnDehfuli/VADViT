# VADViT: Vision Transformer-Driven Malware Detection via Memory Forensics

**VADViT** (Virtual Address Descriptor Vision Transformer) is a novel approach for detecting malicious processes through memory forensics. This repository provides the code for training, testing, and interpreting a Vision Transformer model that leverages Markov, entropy, and VAD-based metadata images to classify malware in volatile memory snapshots.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Training](#2-training)
  - [3. Testing & Explainability](#3-testing--explainability)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [1. VAD Extraction](#1-vad-extraction)
  - [2. Image Generation](#2-image-generation)
  - [3. Vision Transformer](#3-vision-transformer)
  - [4. Explainability](#4-explainability)
- [Results](#results)
- [Future Directions](#future-directions)
- [Citation](#citation)
- [License](#license)

---

## Overview

Modern malware often utilizes sophisticated code injection and obfuscation tactics, making detection in static binaries increasingly challenging. **VADViT** addresses this gap by analyzing **Virtual Address Descriptor (VAD) regions** from memory dumps. It transforms these memory segments into images that reveal hidden malicious patterns, then classifies them using a **Vision Transformer** architecture.

**Paper reference:**  
Dehfouli, Y., & Lashkari, A. H. (2025). *VADViT: A Novel Vision Transformer-Driven Malware Detection Approach for Malicious Process Detection and Explainable Threat Attribution Using Virtual Address Descriptor Regions.* (Preprint)

---

## Key Features

1. **Memory Forensics Focus**: Analyzes VAD regions from memory dumps, capturing stealthy code injections, encrypted payloads, and fileless threats.  
2. **Multi-Modal Image Generation**: Fuses:
   - **Markov images** (byte-transition probabilities),
   - **Entropy images** (randomness/encryption),
   - **Custom intensity images** (protection flags, VAD tags, private pages).
3. **Vision Transformer (ViT)**:
   - Leverages self-attention to model spatial relationships between VAD regions.
   - Achieves robust performance in both **binary** and **multi-class** settings.
4. **Explainability**:
   - Highlights highly attended VAD patches.
   - Maps suspicious memory regions back to original addresses for deeper forensic analysis.
5. **Dataset Logging**: References the *BCCC-MalMem-SnapLog-2025* dataset with memory dumps at 30-second intervals, capturing process-specific PIDs and system events.

---

## Dataset

The paper introduces **BCCC-MalMem-SnapLog-2025**, a dataset containing periodic memory dumps for malicious and benign processes:
- **Memory snapshots** collected every 30 seconds (up to 5 dumps per sample).
- **PID tracking** to isolate each process without sandbox overhead.
- **Network & system event logs** included for cross-referencing.

Because this dataset is not publicly included here, please refer to the paper or contact the authors for details on obtaining or reproducing it.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YaCnDehfuli/VADViT.git
   cd VADViT
   ```

2. **Install Dependencies** (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
   Main libraries include:
   - `torch`, `timm`, `numpy`, `pillow`, `scikit-learn`, `matplotlib`, `opencv-python`

---

## Usage

### 1. Data Preparation
- **VAD Extraction**: Use [Volatility 2 or 3](https://www.volatilityfoundation.org/) (`vadinfo`) to extract VADs from each memory dump.  
- **Image Generation**: Convert each VAD into Markov, entropy, and intensity images. Aggregate them as an RGB image (one channel per feature).  
- **Directory Structure**: Organize generated images into train/val/test splits.

> *Note:* Example scripts or references are provided in the code to guide you in generating these images, though the dataset itself is not bundled.

### 2. Training
Train the model on your prepared dataset:
```bash
python train.py
```
- **Key training parameters** (see `config.py`):
  - `NUM_CLASSES` (2 for binary or more for multi-class).
  - `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`.
  - `MODEL_NAME` (e.g., `vit_base_patch16_224` or `vit_base_patch32_224`).

### 3. Testing & Explainability
Evaluate on test splits and optionally visualize attention maps:
```bash
python test.py --explain
```
- **ROC & Confusion Matrix** are displayed automatically.
- Use `--explain` to overlay attention heatmaps on suspicious VAD patches.

---

## Project Structure

```
VADViT
├── config.py             # Global configs (paths, hyperparams, etc.)
├── train.py              # Training script with progressive layer unfreezing
├── test.py               # Inference and attention overlay
├── dataset_loader.py     # VAD dataset loader class
├── att_visualization.py  # Overlay attention heatmaps for debug/explainability
├── metrics_visualization.py
├── training_utils.py
├── test_utils.py
└── ...
```

---

## How It Works

### 1. VAD Extraction
1. Collect memory dumps at fixed intervals (e.g., 30s).
2. Run Volatility’s `vadinfo` to list each region (`StartVPN, EndVPN`), capturing metadata (protection flags, VAD tags, private memory, etc.).

### 2. Image Generation
1. **Markov Image**: 256×256 matrix of byte transition probabilities (scaled, enhanced).
2. **Entropy Image**: Local Shannon entropy of fixed-size windows across the region.
3. **Intensity Image**: Encodes VAD tag (VadF, VadS, etc.), protection flags (PAGE_EXECUTE, etc.), and private memory in distinct intensities.

These three grayscale images are stacked into **RGB**:
- **R** = intensity (protection, VAD tag, private memory),
- **G** = entropy,
- **B** = markov transitions.

### 3. Vision Transformer
- Splits the final RGB image into fixed-size patches (e.g., 16×16 or 32×32).
- Uses multi-head self-attention to capture relationships among VAD patches.
- Outputs class probabilities (benign vs. malicious, or multi-class family).

### 4. Explainability
- Register attention hooks to retrieve the final self-attention layer.
- Overlay attention onto the original image to pinpoint suspicious VAD offsets.
- Facilitates deeper forensic triage: which VAD addresses contain potential shellcode?

---

## Results

- **Binary Classification**: Achieves up to **99%** accuracy on certain configurations, with an AUC of **0.993**.
- **Multi-Class**: Scores up to **92%** accuracy across 8 malware families + benign.  
- **Explainability**: Case studies show the top-attended region contains self-decoding shellcode.

A typical pipeline run involves:
1. Extracting 5 memory dumps (30s intervals) of a suspicious process.
2. Generating VAD-based Markov, entropy, and intensity images.
3. Inferring with VADViT.
4. Investigating the attention-heavy patches.

---

## Future Directions

- **Dynamic Unfreezing**: Explore more adaptive layer freezing/unfreezing to handle small datasets better.
- **Hybrid Models**: Merge other backbones (Swin Transformers, ResNets) for multi-scale VAD features.
- **More VAD Analysis**: Compare enumerated VAD structures with page table entries (PTEs) to detect deeper code manipulation or DKOM-based stealth injections.
- **Larger-Scale Datasets**: BCCC-MalMem-SnapLog-2025 can be expanded or integrated with publicly available memory corpora to enhance coverage of advanced malware families.

---

## Citation

If you use VADViT or the concepts in your research, please cite:

```
@misc{dehfouli2025VADViT,
  title={VADViT: A Novel Vision Transformer-Driven Malware Detection Approach for Malicious Process Detection and Explainable Threat Attribution Using Virtual Address Descriptor Regions},
  author={Dehfouli, Y. and Lashkari, A.H.},
  year={2025},
  note={Preprint}
}
```

---

## License
This project is licensed under the [MIT License](LICENSE). Feel free to modify and distribute with attribution.

---

**Questions, suggestions, or issues?**  
Open an [issue](https://github.com/YaCnDehfuli/VADViT/issues) or contact the authors. Happy hunting!
