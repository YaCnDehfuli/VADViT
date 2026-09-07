# Input and output configuration

All paths accept `~`. Relative paths resolve from the repository root, including
when a preprocessing command is run from its own directory. Input directories
must contain your authorized dataset; memory dumps and trained checkpoints are
not included in the checkout.

| Environment variable | Use | Default below the repository root |
| --- | --- | --- |
| `VADVIT_DUMPS_DIR` | Family/sample directories containing `Dumps/` | `data/BCCC-Mal-NetMem-2025-Trojan-Onwards` |
| `VADVIT_REGIONS_DIR` | Extracted per-snapshot regions | `data/BCCC_Dataset` |
| `VADVIT_CONSOLIDATED_DIR` | Shared output/input between preprocessing stages | `data/BCCC_Consolidated_Dataset` |
| `VADVIT_VOLATILITY` | Path to the Volatility `vol.py` entry script | `volatility3/vol.py` |
| `VADVIT_IMAGE_DATASETS_DIR` | Generated image root and single-sample image root | `data/Image_Datasets` |
| `VADVIT_DATASET_PATH` | Dataset used by training/evaluation | `data/{PATCH_SIZE}_{IMAGE_SIZE}_Datasets/{PATCH_SIZE}_{IMAGE_SIZE}_{MODE}` |
| `VADVIT_CHECKPOINT` | Model checkpoint output/input | `models/{MODE}_{PATCH_SIZE}_{IMAGE_SIZE}_{FROZEN_LAYERS}f_{STEPS}u.pt` |
| `VADVIT_AUC_FOLDER` | ROC figure output | `outputs/AUCs` |
| `VADVIT_CM_FOLDER` | Confusion-matrix output | `outputs/CMs` |

Set `VADVIT_VOLATILITY` to your Volatility entry script. Run preprocessing with
the same Python environment as the root commands. For image construction, choose
matching `PATCH_SIZE` and `IMAGE_SIZE` in the grid configuration and root model
configuration. Set `VADVIT_DATASET_PATH` to the prepared binary or family dataset
for the selected `MODE`; the dataset layout must match that mode.

The root defaults select the family model with image size 224 and patch size 32.
The grid script retains its historical 384/16 defaults; select the model's image
and patch dimensions before constructing grids. Changing these settings is not
evidence of reproducing a published score.

By default, dump preprocessing visits every sample in sorted family/sample
order. To resume deliberately, pass `--resume-from HASH`; that sample is included.
A missing resume hash is an error. The externally referenced directory name
`Data Preprocessing/Dumps_to_Cnosolidated` is retained.

Single-sample inspection requires explicit family and sample identifiers:

```bash
python sample_test.py --help
python sample_test.py FAMILY SAMPLE_HASH
```

The image path is `{image-root}/{PATCH_SIZE}_{IMAGE_SIZE}/{family}/{sample_hash}.png`.
Region numbering remains contiguous: executable regions first, then DLL regions.
