# DiSPA

Differential Substructure-Pathway Attention for Drug Response Prediction
<img width="760" height="380" alt="스크린샷 2026-01-12 오후 7 15 32" src="https://github.com/user-attachments/assets/2eeb1680-0307-44af-88a5-0b24163b3b85" />
## Project Structure

```
DiSPA/
├── model.py          # DrugResponseModel
├── train.py          # Training (5-fold CV)
├── test.py           # Evaluation
├── dataset.py        # Dataset & DataLoader

├── split.py          # Data splitting
├── utils.py          # Utilities
├── config.yml        # Hyperparameters
├── environment.yml   # Dependencies
│
├── modules/
│   ├── cross_attn.py       # Cross-attention
│   ├── diff_cross_attn.py  # Differential cross-attention
│   ├── ffn_layer.py        # FFN layers
│   ├── mlp_head.py         # Regressor head
│   └── rms_norm.py         # RMS normalization
│
├── input/            # Embeddings & pathway indices
└── Sample_Data/      # Train/val/test splits
```

## Installation

```bash
conda env create -f environment.yml
conda activate dispa
```

## Training

```bash
python train.py
```

Results saved to `checkpoints/`, `plots/`, `log/`

## Testing

```bash
python test.py --checkpoint_date 20250712_21
```

Results saved to `results/`

## Configuration


Edit `config.yml` to change hyperparameters:

```yaml
training:
  batch_size: 2048
  learning_rate: 0.0004
  num_epochs: 300

model:
  isDiffer: true      # Use differential attention
  num_heads: 4
  depth: 1
```
**Note:** Full hyperparameters will be updated soon.
