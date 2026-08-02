# VADViT

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/ML-PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Volatility 3](https://img.shields.io/badge/Forensics-Volatility_3-111827)](https://volatilityfoundation.org/)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jisa.2025.104200-0077B5)](https://doi.org/10.1016/j.jisa.2025.104200)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)

**Technical focus:** memory forensics · malicious-process detection · Vision Transformers · VAD analysis · explainable AI

VADViT is a memory-forensics research pipeline for malicious-process
detection from Virtual Address Descriptor (VAD) regions. It extracts process
memory regions from interval memory snapshots, converts those regions into
Markov, entropy, and VAD-metadata image channels, arranges them into a
process-level grid, and trains a Vision Transformer for binary or malware-family
classification.

The implementation accompanies the paper:

> Yasin Dehfouli and Arash Habibi Lashkari, "VADViT: Vision
> transformer-driven memory forensics for malicious process detection and
> explainable threat attribution," Journal of Information Security and
> Applications, 94, 104200, 2025. DOI: `10.1016/j.jisa.2025.104200`.

The paper reports 99.2% binary accuracy for the best VADViT configuration and
92% macro-averaged F1 for multiclass family classification on
BCCC-MalMem-SnapLog-2025. Those numbers depend on the dataset, split, training
configuration, and checkpoints used in the study; this repository does not ship
the raw memory dumps or trained weights.


![VADViT workflow (Published to JISA : `10.1016/j.jisa.2025.104200`)](docs/assets/jisa-workflow.png)

## What This Repository Contains

- Volatility-based VAD extraction from per-sample memory snapshots.
- Snapshot consolidation that keeps the richest process-memory view for each
  sample.
- Region categorization into executable/malware, DLL-backed, and heap/stack
  groups.
- Markov, entropy, and intensity-channel image generation for each retained VAD
  region.
- Process-level grid construction for ViT inputs.
- ViT training, validation, test evaluation, and attention-overlay utilities.

## Repository Layout

```text
Data Preprocessing/
  Dumps_to_Cnosolidated/       Volatility vadinfo extraction, region division,
                               and snapshot consolidation
  Consolidated_to_Grid/        VAD region -> RGB patch -> process grid images
dataset/                       ImageDataset and train/val/test transforms
models/                        timm ViT wrapper with configurable frozen blocks
utils/                         training loop, metrics, attention visualization
config.py                      training/evaluation configuration
train.py                       train a ViT on generated grid images
test.py                        evaluate a saved model, optionally with attention
sample_test.py                 single-image inspection helper with hard-coded paths
requirements.txt               Python dependency pins used by the project
```

The directory name `Dumps_to_Cnosolidated` is intentionally documented as it
exists in the repository, typo included, so commands can be copied directly.

## Data Requirements

Raw dumps are not committed to this repository. The expected preprocessing input
is organized by malware family and sample hash:

```text
BASE_DIR/
  Trojan/
    <sample_sha256>/
      Dumps/
        <pid>_snapshot1.vmem
        <pid>_snapshot2.vmem
        ...
  Benign/
    <sample_id>/
      Dumps/
        <pid>_snapshot1.vmem
```

Each dump filename must start with the target PID because
`region_extractor.py` reads the PID from `dump_file.split("_")[0]`. The original
study used up to five interval snapshots per sample. Samples where the PID is no
longer present in `windows.pslist` are stopped early; samples with only one dump
are counted separately because they contain less temporal evidence.

## Environment

Use a Python environment that matches your CUDA/PyTorch setup. The committed
requirements file pins the repository's working dependency set; the paper's
reported experiments should still be reproduced with the exact environment used
for that run. GPU users may need to install the PyTorch build appropriate for
their driver before installing the remaining packages.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Volatility 3 is required for the extraction stage. Set its path in
`Data Preprocessing/Dumps_to_Cnosolidated/config.py`.

## Stage 1: Extract And Consolidate VAD Regions

Edit `Data Preprocessing/Dumps_to_Cnosolidated/config.py`:

```python
BASE_DIR = "/path/to/BCCC-MalMem-SnapLog-2025"
OUTPUT_DIR = "/path/to/intermediate_vad_regions"
CONSOLIDATED_DIR = "/path/to/consolidated_vad_regions"
VOLATILITY = "/path/to/volatility3/vol.py"
```

Run the extraction pipeline:

```bash
cd "Data Preprocessing/Dumps_to_Cnosolidated"
python main.py
```

This stage performs three operations for each sample:

1. Runs `windows.pslist` to confirm the target PID exists in each snapshot.
2. Runs `windows.vadinfo --pid <pid> --dump` and writes `vadinfo.csv` plus dumped
   VAD region files.
3. Sorts retained regions into `malware_executable/`, `dlls/`, and
   `heap_and_stack/`, then copies the snapshot with the most executable/DLL
   regions into `CONSOLIDATED_DIR`.

The consolidated output is expected to look like this:

```text
CONSOLIDATED_DIR/
  Trojan/
    <sample_sha256>/
      malware_executable/
        malware_executable_regions.csv
        vad.0x...dmp
      dlls/
        dll_regions.csv
        vad.0x...dmp
```

## Stage 2: Build Process Grid Images

Edit `Data Preprocessing/Consolidated_to_Grid/config.py`:

```python
IMAGE_SIZE = 224       # paper evaluates 224 and 384
PATCH_SIZE = 32        # paper evaluates 16 and 32
CONSOLIDATED_DIR = "/path/to/consolidated_vad_regions"
IMAGE_DATASET_DIR = "/path/to/image_datasets"
```

Run grid generation:

```bash
cd "Data Preprocessing/Consolidated_to_Grid"
python main.py
```

For each retained VAD region, `process2image.py` creates:

- red channel: VAD tag and protection metadata intensity;
- green channel: dynamic-window Shannon entropy;
- blue channel: Markov byte-transition structure.

Executable regions are placed first in ascending VAD address order, followed by
DLL-backed regions. Empty grid cells are zero-padded. The output path is:

```text
IMAGE_DATASET_DIR/
  32_224/
    Trojan/
      <sample_sha256>.png
    Benign/
      <sample_id>.png
```

## Stage 3: Train VADViT

Edit the root `config.py`:

```python
IMAGE_SIZE = 224
PATCH_SIZE = 32
MODE = "Binary"        # "Binary" or "Multi"
FROZEN_LAYERS = 6
STEPS = 3
DATASET_PATH = "/path/to/image_datasets/32_224/32_224_Binary"
SAVE_PATH = "./models/Binary_32_224_6f_3u.pt"
```

Then run:

```bash
python train.py
```

The dataset loader creates an 80/10/10 train/validation/test split from class
folders under `DATASET_PATH`. For binary mode, `Benign` maps to class `0` and
every other folder maps to class `1`. For multiclass mode, folders are sorted
alphabetically and mapped to numeric labels.

Training uses a timm ViT backbone, label-smoothed cross-entropy,
ReduceLROnPlateau, gradual unfreezing of frozen transformer blocks, temperature
scaling during prediction, and late stochastic weight averaging. The best model
is saved to `SAVE_PATH`; the metric plot is saved as `training_plot.png`.

## Evaluation And Explainability

Evaluate a saved model:

```bash
python test.py
```

Enable last-block attention visualization:

```bash
python test.py --explain
```

`test.py` loads the `test` split from `DATASET_PATH`, prints a classification
report and confusion matrix, and writes ROC/confusion-matrix plots to the folders
configured by `AUC_FOLDER` and `CM_FOLDER`. The ROC helper is designed around
binary scoring, so use it carefully when `MODE = "Multi"`.

For one-off inspection, edit the hard-coded paths in `sample_test.py` and run:

```bash
python sample_test.py
```

That helper loads one process-grid image, prints class probabilities, displays
an attention overlay, and lists the executable/DLL VAD region files so the patch
order can be traced back to addresses.

## Paper Configuration Notes

The paper evaluates combinations of image size, patch size, and frozen-layer
strategy. The strongest binary configuration was `PATCH_SIZE=32`,
`IMAGE_SIZE=224`, `FROZEN_LAYERS=6`, and `STEPS=3`. The multiclass experiment
uses the same `32_224_6f` family-label setting as one of the two final
configurations.

The paper's dataset, BCCC-MalMem-SnapLog-2025, includes malware samples from
Backdoor, Exploit, HackTool, Hoax, Rootkit, Trojan, Virus, and Worm families,
plus benign samples. Raw dumps are omitted here due to size and handling
constraints.

## Important Boundaries

- VADViT analyzes memory regions captured from a target process; it is not a
  live EDR or antivirus product.
- Attention maps are forensic leads, not proof of causality.
- Reported metrics require the original dataset, split discipline, and training
  setup; do not reuse them for a different corpus without re-evaluation.
- Volatility symbol support and dump quality directly affect region extraction.

## Citation

```bibtex
@article{dehfouli2025vadvit,
  title = {VADViT: Vision transformer-driven memory forensics for malicious process detection and explainable threat attribution},
  author = {Dehfouli, Yasin and Lashkari, Arash Habibi},
  journal = {Journal of Information Security and Applications},
  volume = {94},
  pages = {104200},
  year = {2025},
  doi = {10.1016/j.jisa.2025.104200}
}
```

## License

[MIT](LICENSE). Dataset access and third-party tools used with the pipeline may
have separate terms. If this repository supports your work, please cite the
paper using [`CITATION.cff`](CITATION.cff).
