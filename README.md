# VADViT

VADViT classifies Windows process memory with a Vision Transformer and ranks the
memory regions that contributed most to each verdict.

[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jisa.2025.104200-0077B5)](https://doi.org/10.1016/j.jisa.2025.104200)
[![Release](https://img.shields.io/github/v/release/YaCnDehfuli/VADViT)](https://github.com/YaCnDehfuli/VADViT/releases)

## Results

Published in the *Journal of Information Security and Applications*, vol. 94,
art. 104200 (2025).

| Task                                   | Metric              | Score |
| -------------------------------------- | ------------------- | ----- |
| Malicious vs. benign process detection | Accuracy            | 99.2% |
| Malware family attribution             | Macro-averaged F1   | 92%   |

These results are from BCCC-MalMem-SnapLog-2025. Attention-based ranking orders
VAD regions by their contribution to the verdict, reducing the set an analyst
must inspect manually.

<img src="docs/figures/architecture.svg" alt="VADViT architecture" width="880">

**Research artifact.** This repository is the reference implementation for the
published paper. It is not an endpoint agent, detection product, or live
monitoring tool.

## Quickstart

A clean checkout includes neither memory dumps nor trained weights. The commands
below install the pinned environment and verify that the model and test entry
point import correctly. They do not train the model or reproduce the published
scores.

Use CPython 3.10. The pinned environment uses `torch==2.1.0`,
`torchvision==0.16.0` and `timm==0.6.12`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "from models.ViT_model import ViTForImages; from config import MODEL_NAME, MODE, NUM_CLASSES; print(MODEL_NAME, MODE, NUM_CLASSES)"
python test.py --help
```

## How it works

`Data Preprocessing/Dumps_to_Cnosolidated/region_extractor.py` traverses each
sample's `Dumps/` directory. It reads the PID from the `.vmem` filename prefix
before the first underscore. For each snapshot that still lists the PID, the
extractor runs Volatility 3 as `windows.vadinfo --pid=<pid> --dump`;
`windows.pslist` serves only as a presence check. Samples with a single dump are
marked `timeout` and skipped.

`region_divider.py` classifies dumped regions as `malware_executable/`, `dlls/`,
or `heap_and_stack/` from the protection and mapped-file fields in `vadinfo.csv`.
`dump_selector.py` copies the snapshot with the largest combined count of
executable and DLL regions into the consolidated tree. Grid construction uses
only `malware_executable` and `dlls`. Keep the committed
`Dumps_to_Cnosolidated` spelling.

`Data Preprocessing/Consolidated_to_Grid/process2image.py` converts each retained
`.dmp` file into a square RGB patch. Red encodes a constant VAD tag plus
protection intensity; green encodes windowed Shannon entropy (the committed grid
config sets `ENT_METHOD = DYNAMIC`); blue encodes a downsampled Markov
byte-transition matrix. The pipeline sorts patches by the `vad.0x...` address
in the filename, places executable regions before DLL regions, and assembles a
process-level grid. It zero-pads empty cells and truncates overflow to the grid
capacity.

`models/ViT_model.py` loads the pretrained `timm` Vision Transformer selected by
`MODEL_NAME` (`vit_base_patch{PATCH_SIZE}_{IMAGE_SIZE}`) and replaces its
classification head. `MODE` in `config.py` selects two classes for `Binary` or
nine for `Multi`. The training loop in `utils/training_utils.py` freezes the
first `FROZEN_LAYERS` blocks, progressively unfreezes them over `STEPS`, uses
label-smoothed cross-entropy and temperature-scaled softmax (`T = 0.7`), and
starts stochastic weight averaging at epoch 32.

`test.py --explain` and `sample_test.py` register a forward hook on the final
attention block, extract the class-token row, and render the overlay through
`utils/att_visualization.py`. Because every grid cell maps to one VAD region
file, an analyst can inspect the highest-attention addresses first.

<img src="docs/figures/attention-overlay.png" alt="Attention overlay" width="880">

*Published Fig. 16: attention overlaid on a process-level VAD grid. Left: a
strongly attended cell the paper describes as indicative of malicious behavior.
Right: a moderately attended cell. The paper does not name a PID or virtual
address on this figure. Ranked cells are the subset an analyst inspects first.*

## Reproducing the published results

The paper evaluates BCCC-MalMem-SnapLog-2025. This repository does not include
the raw dumps or checkpoints. The paper's data-availability statement says the
captured dumps are available to academic researchers on request under the name
BCCC-Mal-NetMem-2025 and a non-commercial academic licence. The dataset should
not be treated as a public download.

Set the input and output locations using the [configuration guide](docs/configuration.md).
The preprocessing stages share the same consolidated-region location.

With the required inputs configured:

```bash
cd "Data Preprocessing/Dumps_to_Cnosolidated"
python main.py
cd "../Consolidated_to_Grid"
python main.py
cd ../..
python train.py
python test.py
python test.py --explain
```

On completion, the run writes a checkpoint to `SAVE_PATH`,
`training_plot.png`, and ROC and confusion-matrix PDFs under `AUC_FOLDER` and
`CM_FOLDER`. `sample_test.py` prints class probabilities, displays an overlay,
and lists executable-region filenames followed by DLL-region filenames in
address-sortable order.

The paper's training description differs from the current implementation in two
places. It specifies AdamW with weight decay; the exact weight-decay value has not
yet been verified against the paper. `train.py` uses `optim.Adam`. It also reports a
`ReduceLROnPlateau` factor of `0.5`, while this tree uses `0.33`. These
differences remain unresolved. `dataset/dataset_loader.py` creates an 80/10/10
split with seed `42`, and `train.py` aborts if the splits share image paths.

## Limitations

- Results are specific to BCCC-MalMem-SnapLog-2025 and its capture methodology.
- VAD extraction runs only for processes that appear in the active process list
  at snapshot time. An analyst still has to try candidate PIDs before the
  matching VAD pattern is found.
- The 30-second snapshot cadence can miss short-lived injections or memory
  wipes. Samples that unmap, encrypt, or repurpose VAD regions between
  snapshots can slip past.
- The method needs a full RAM dump and the live PID so Volatility can rebuild
  the VAD tree. Feature-only sets such as CIC-MalMem and process-only
  collections such as Dumpware10 cannot be used as external test beds.
- Trojan family attribution is weaker than other classes. The paper reports
  lower Trojan recall and treats Trojan as a catch-all when families share
  injection stubs, packers, encryption layers, and high-entropy allocations.
- Sparse single-snapshot captures make Exploit and Backdoor look like Trojan
  loader stubs. The confusion-matrix discussion cites that overlap as a cause
  of those swaps.
- The committed extractor targets 64-bit Windows images. Linux and macOS dumps
  are listed as later work, not as supported inputs.
- ViT-Base training and inference memory is too heavy for many endpoint or IoT
  devices without a discrete GPU. The paper states that quantization or a
  distilled backbone would be required for those cases.

## Citation

Crossref record for `10.1016/j.jisa.2025.104200` (preferred):

```bibtex
@article{Dehfouli2025VADViT,
  title   = {VADViT: Vision transformer-driven memory forensics for malicious process detection and explainable threat attribution},
  author  = {Dehfouli, Yasin and Lashkari, Arash Habibi},
  journal = {Journal of Information Security and Applications},
  volume  = {94},
  pages   = {104200},
  year    = {2025},
  month   = nov,
  doi     = {10.1016/j.jisa.2025.104200}
}
```

## Related work in this portfolio

Memory forensics → detection engineering → evaluation of AI in security operations.

| Repository | What it establishes |
| --- | --- |
| [VolMemLyzer3](https://github.com/YaCnDehfuli/VolMemLyzer3-CLI_forensic_tool) | Volatility 3 orchestration and feature extraction; 2.4× parallel speedup on a pinned 10-plugin set |
| **VADViT** | Published ViT classification of process memory — 99.2% binary accuracy, 92% macro-F1 |
| [MalGraph](https://github.com/YaCnDehfuli/MalGraph) | Why memory-time recovery matters: UPX packing leaves 5.4% of functions statically recoverable |
| [MemTriage](https://github.com/YaCnDehfuli/MemTriage) | The analyst workspace that consumes both |
| [detection-under-load](https://github.com/YaCnDehfuli/detection-under-load) | Published Sigma coverage for T1003.001 collapses under operator-controlled renaming |
| [agent-under-load](https://github.com/YaCnDehfuli/agent-under-load) | Whether an LLM agent can triage those detections, measured against deterministic ground truth |
