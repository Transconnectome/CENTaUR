# Ko-CENTaUR Setup Guide

Complete setup guide for Korean language adaptation of CENTaUR cognitive modeling system.

## Environment Setup

### Server Specifications
- **GPU**: 8× NVIDIA GeForce RTX 3090 (24GB each)
- **CUDA**: 12.4
- **Storage**: /scratch/connectome/connectome1/ko-centaur

### Python Environment

```bash
# Create conda environment
conda create -n ko-centaur python=3.10
conda activate ko-centaur

# Install PyTorch 2.5.1 with CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install ML packages
pip install transformers==4.57.0 \
    datasets \
    peft \
    accelerate \
    bitsandbytes \
    scipy \
    scikit-learn \
    pandas \
    "numpy<2"
```

### Environment Variables

Add to `~/.bash_profile`:

```bash
# Ko-CENTaUR Environment
export SCRATCH_BASE="/scratch/connectome/connectome1"
export WORK_DIR="$SCRATCH_BASE/ko-centaur"
export TMPDIR="$WORK_DIR/tmp"
export PIP_CACHE_DIR="$WORK_DIR/pip-cache"
export HF_HOME="$WORK_DIR/cache"
export TORCH_HOME="$WORK_DIR/models"
export TRANSFORMERS_CACHE="$WORK_DIR/cache"

# Auto-activate conda environment
if [ -f "$SCRATCH_BASE/miniconda3/bin/activate" ]; then
    . "$SCRATCH_BASE/miniconda3/bin/activate" ko-centaur
fi
```

## Model Setup

### EXAONE-3.0-7.8B-Instruct

1. **Request Access**: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct

2. **HuggingFace Login**:
```bash
huggingface-cli login --token YOUR_TOKEN
```

3. **Test Model Loading**:
```bash
python scripts/test_exaone.py
```

Expected output:
- Tokenizer: 102,400 vocab size
- Model: 4-bit quantized, ~4.8 GB GPU memory
- Generation: Korean language response successful

## Data Pipeline

### Psych-101 Dataset

```bash
# Download dataset
python scripts/download_psych101.py

# Explore data structure
python scripts/explore_data.py

# Preprocess for EXAONE format
python scripts/preprocess_psych101.py
```

**Dataset Statistics**:
- Total samples: 60,092
- Format: EXAONE instruction format
- Size: 833.5 MB

## Training

### QLoRA Fine-tuning

```bash
# Test training (50 samples)
python scripts/train_exaone_qlora.py
```

**Training Configuration**:
- LoRA rank: 8, alpha: 16
- 4-bit quantization (NF4)
- Batch size: 1 × 4 gradient accumulation
- Optimizer: paged_adamw_8bit
- Learning rate: 2e-4

**GPU Usage**:
- Use GPU 2, 6, or 7 (check availability with `nvidia-smi`)
- Set `CUDA_VISIBLE_DEVICES` environment variable

**Expected Results** (50 samples test):
- Training time: ~60 seconds
- Loss reduction: 1.2564 → 0.7185 (43% decrease)
- Speed: ~4.8 seconds/step

## Troubleshooting

### Multi-GPU Device Error

If you encounter "Expected all tensors to be on the same device" error:

1. Check GPU availability: `nvidia-smi`
2. Use empty GPU by setting: `export CUDA_VISIBLE_DEVICES="2"`
3. Ensure Transformers >= 4.57.0

### Out of Memory

If GPU memory is insufficient:
- Reduce `max_length` (default: 512)
- Reduce `batch_size` (default: 1)
- Increase `gradient_accumulation_steps`
- Use a different empty GPU

### Connection Issues

If SSH connection resets:
- Use screen for long-running tasks: `screen -S ko-centaur`
- Detach: `Ctrl+A, D`
- Reattach: `screen -r ko-centaur`

## Directory Structure

```
/scratch/connectome/connectome1/ko-centaur/
├── cache/              # HuggingFace cache
├── data/               # Datasets
│   ├── psych101_train.jsonl
│   └── psych101_exaone_train.jsonl
├── models/             # Trained models
│   └── test_gpu2/final/
├── tmp/                # Temporary files
└── pip-cache/          # pip cache
```

## Verification Checklist

- [ ] Conda environment activated
- [ ] PyTorch 2.5.1+cu121 installed
- [ ] Transformers 4.57.0 installed
- [ ] EXAONE model loads successfully
- [ ] Psych-101 dataset downloaded
- [ ] Test training completes without errors
- [ ] Model generates Korean responses

## Next Steps

1. **Scale Up Training**: Use full 60K samples
2. **Korean Data**: Integrate Korean psychological experiment data
3. **Multi-Lab Collaboration**: Establish data sharing agreements
4. **Evaluation**: Implement held-out testing and zero-shot evaluation
