# InstaNovo-P: A De Novo Peptide Sequencing Model for Phosphoproteomics

InstaNovo-P is a phosphorylation-specific version of the transformer-based [InstaNovo](https://github.com/instadeepai/InstaNovo) model, fine-tuned on extensive phosphoproteomics datasets. It significantly surpasses existing methods in phosphorylated peptide detection and phosphorylation site localization accuracy across multiple datasets.

**Paper:** [InstaNovo-P: A de novo peptide sequencing model for phosphoproteomics](https://doi.org/10.1101/2025.05.14.654049) (bioRxiv preprint)

> **Important -- Codebase Lineage:**
> This repository contains a self-contained fork of [InstaNovo v0.1.6](https://pypi.org/project/instanovo/0.1.6/) with substantial modifications for phosphoproteomics fine-tuning. It is **not** compatible with newer InstaNovo releases (>=1.0).
>
> **For inference only**, use the latest [`instanovo`](https://github.com/instadeepai/InstaNovo) package (>=1.1.2) with the released [InstaNovo-P checkpoint](https://github.com/instadeepai/InstaNovo/releases/download/1.1.2/instanovo-phospho-v1.0.0.ckpt). See the [Quick Start -- Inference](#quick-start----inference) section below.
>
> This repository provides the **training and fine-tuning code** to reproduce the results presented in the paper.

## Table of Contents

- [Quick Start -- Inference](#quick-start----inference)
- [Datasets](#datasets)
- [Codebase Lineage and Modified Files](#codebase-lineage-and-modified-files)
- [Reproducing the Fine-Tuning](#reproducing-the-fine-tuning)
- [Evaluation](#evaluation)
- [Hyperparameters](#hyperparameters)
- [Citation](#citation)
- [License](#license)

## Quick Start -- Inference

For running inference with InstaNovo-P, we recommend using the latest `instanovo` package which provides a streamlined interface:

```bash
pip install instanovo>=1.1.2
```

The InstaNovo-P checkpoint is available at:
- [instanovo-phospho-v1.0.0.ckpt](https://github.com/instadeepai/InstaNovo/releases/download/1.1.2/instanovo-phospho-v1.0.0.ckpt)

For a step-by-step inference tutorial, see the official notebook:
- [InstaNovo-P.ipynb](https://raw.githubusercontent.com/instadeepai/InstaNovo/refs/heads/main/notebooks/InstaNovo-P.ipynb)

The notebook demonstrates:
- Loading the InstaNovo-P checkpoint
- Loading phosphoproteomics data from HuggingFace
- Running inference with greedy, beam search, and knapsack beam search decoding
- Evaluating prediction confidence and computing metrics

## Datasets

### Training Data

The InstaNovo-P training dataset is available on HuggingFace:
- [InstaDeepAI/InstaNovo-P](https://huggingface.co/datasets/InstaDeepAI/InstaNovo-P)

This dataset comprises 2.57 million phosphorylated peptide-spectrum matches (PSMs) from 29 PRIDE projects, reprocessed with IonBot (filtered at 0.80 localization probability) and split using GraphPart homology partitioning into train/validation/test sets.

### Evaluation Datasets

- **FGFR2 Validation:** `[PLACEHOLDER: HuggingFace link TBD]`
- **Astral Benchmarking:** `[PLACEHOLDER: HuggingFace link TBD]`

## Codebase Lineage and Modified Files

This repository is based on [InstaNovo v0.1.6](https://pypi.org/project/instanovo/0.1.6/) with extensive modifications for phosphoproteomics fine-tuning. Below is a summary of every file and its changes relative to the public `instanovo==0.1.6` release.

### Modified Files

| File | Diff Size | Description |
|------|-----------|-------------|
| `instanovo/transformer/train.py` | 610 lines | Core training loop rewritten for phospho fine-tuning. Adds gradual unfreezing via `FinetuningScheduler`, slanted triangular learning rate scheduling (`STLRScheduler`), PTM-specific metric logging (phospho precision/recall), checkpoint key remapping for the extended residue vocabulary, and Polars IPC / HuggingFace data loading. |
| `instanovo/transformer/predict.py` | 555 lines | Prediction pipeline adapted for phospho residues. Adds PTM residue remapping between internal notation (`S(+79.97)`) and ProForma (`S[UNIMOD:21]`), evaluation on phospho-specific metrics, and flexible checkpoint loading with vocabulary extension support. |
| `instanovo/transformer/model.py` | 368 lines | Model architecture extended with optional LoRA adapter support for parameter-efficient fine-tuning, checkpoint loading logic for vocabulary-extended models (mapping new phospho tokens into the embedding/output layers), and decode helper methods. |
| `instanovo/utils/metrics.py` | 284 lines | Evaluation metrics extended with PTM-specific precision and recall (measuring correct localization of phosphorylation and oxidation sites independently of overall peptide accuracy). |
| `instanovo/utils/residues.py` | 197 lines | Residue vocabulary extended with phosphorylated residues (`S(+79.97)`, `T(+79.97)`, `Y(+79.97)`) and oxidized methionine (`M(+15.99)`). Adds bidirectional mapping between internal and ProForma/UNIMOD notation. |
| `instanovo/transformer/dataset.py` | 79 lines | Dataset class extended with string-return mode for sequence labels, flexible column name mapping, and Polars IPC format support alongside the original MGF-based loading. |
| `instanovo/utils/convert_to_ipc.py` | 52 lines | Data conversion script updated to handle the phospho dataset schema. |
| `instanovo/inference/beam_search.py` | 25 lines | Minor adjustments to beam search decoder. |
| `instanovo/constants.py` | 14 lines | Added `SpecialTokens` enum (PAD, EOS, SOS). |
| `instanovo/inference/knapsack_beam_search.py` | 4 lines | Trivial changes. |

### Unchanged Files

These files are identical to their `instanovo==0.1.6` counterparts:

- `instanovo/transformer/layers.py` -- Transformer encoder/decoder layer definitions
- `instanovo/inference/knapsack.py` -- Knapsack algorithm for constrained beam search
- `instanovo/inference/interfaces.py` -- Abstract interfaces for decoders

### New Files (not in v0.1.6)

| File | Description |
|------|-------------|
| `instanovo/utils/schedulers.py` | Implements `STLRScheduler` and `STLRSchedulerEpochBased` for slanted triangular learning rate scheduling used during gradual unfreezing. |
| `instanovo/utils/minLoRA.py` | Lightweight LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning of transformer layers. |

## Reproducing the Fine-Tuning

### 1. Clone the Repository

```bash
git clone https://github.com/<OWNER>/instanovo-p.git
cd instanovo-p
```

### 2. Install Dependencies

This project uses [uv](https://docs.astral.sh/uv/) for dependency management with Python 3.10 and pinned dependency versions for reproducibility.

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

### 3. Download the Base InstaNovo Checkpoint

The fine-tuning starts from the base InstaNovo model (v0.1.4):

```bash
wget https://github.com/instadeepai/InstaNovo/releases/download/0.1.4/instanovo.pt -P checkpoints/
```

This downloads the 361 MB checkpoint to `./checkpoints/instanovo.pt`.

### 4. Download and Prepare Training Data

Download the training data from HuggingFace:

```python
from datasets import load_dataset

dataset = load_dataset("InstaDeepAI/InstaNovo-P")
```

Convert the data to Polars IPC format for training:

```bash
python -m instanovo.utils.convert_to_ipc --input <path_to_data> --output data/
```

### 5. Run Training

```bash
python -m instanovo.transformer.train \
    --train_path data/train.ipc \
    --valid_path data/valid.ipc \
    --config instanovo_finetune_phospho.yaml \
    --n_workers 8
```

The training configuration is defined in `configs/instanovo/instanovo_finetune_phospho.yaml` and uses the gradual unfreezing schedule from `configs/finetune_scheduler/finetune_schedule_gu_decoder-encoder-v2.yaml`.

The fine-tuning proceeds through 10 epochs with encoder-first gradual unfreezing:
1. **Epochs 0-1:** Head and embedding layers only
2. **Epochs 1-5:** Decoder layers progressively unfrozen (top to bottom)
3. **Epochs 5-10:** Encoder layers progressively unfrozen (top to bottom)

## Evaluation

Run prediction on a test set:

```bash
python -m instanovo.transformer.predict \
    <test_data.ipc> \
    <checkpoint_path> \
    --output-path predictions.csv \
    --n-beams 5 \
    --batch-size 128
```

Evaluation metrics (amino acid precision/recall, peptide precision/recall, PTM-specific precision/recall) are computed during validation and logged to TensorBoard. The predictions CSV can be used for further downstream analysis.

## Hyperparameters

The key hyperparameters used for fine-tuning InstaNovo-P:

| Parameter | Value |
|-----------|-------|
| Base model | InstaNovo v0.1.4 (768-dim, 9 layers, 16 heads) |
| Optimizer | Adam |
| Learning rate scheduler | STLR (Slanted Triangular LR), epoch-based |
| Max learning rate | 2e-4 (head), 1e-5 to 2e-6 (deeper layers) |
| LR ratio | 1/32 |
| Batch size | 64 |
| Epochs | 10 |
| Weight decay | 1e-6 |
| Dropout | 0.1 |
| Gradient clipping | 10.0 |
| Unfreezing strategy | Gradual (decoder-first, then encoder) |
| Validation beams | 2 |
| Training samples | ~1.89M |
| Validation subset | 2% (~4,437 samples) |

## Citation

If you use InstaNovo-P in your research, please cite:

```bibtex
@article {Lauridsen2025.05.14.654049,
  author = {Lauridsen, Jesper and Ramasamy, Pathmanaban and Catzel, Rachel and Canbay, Vahap and Mabona, Amandla and Eloff, Kevin and Fullwood, Paul and Ferguson, Jennifer and Kirketerp-M{\o}ller, Annekatrine and Goldschmidt, Ida Sofie and Claeys, Tine and van Puyenbroeck, Sam and Lopez Carranza, Nicolas and Schoof, Erwin M. and Martens, Lennart and Van Goey, Jeroen and Francavilla, Chiara and Jenkins, Timothy Patrick and Kalogeropoulos, Konstantinos},
  title = {InstaNovo-P: A de novo peptide sequencing model for phosphoproteomics},
  elocation-id = {2025.05.14.654049},
  year = {2025},
  doi = {10.1101/2025.05.14.654049},
  publisher = {Cold Spring Harbor Laboratory},
  URL = {https://www.biorxiv.org/content/early/2025/05/18/2025.05.14.654049},
  eprint = {https://www.biorxiv.org/content/early/2025/05/18/2025.05.14.654049.full.pdf},
  journal = {bioRxiv}
}
```

> **Note:** This citation refers to the bioRxiv preprint. It will be updated once the peer-reviewed publication is available.

## License

This project is licensed under the Apache License 2.0 -- see the [LICENSE](LICENSE) file for details.
