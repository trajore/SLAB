# BERT Gradual Training with Quad/2Quad Functions

This directory contains a modified version of the BERT gradual training that supports:

1. **Quad activation functions**: Replaces GELU with quadratic activations
2. **2Quad softmax**: Custom softmax with quadratic transformation  
3. **Pre-trained model loading**: Use existing quad-trained models as starting points
4. **Gradual LayerNorm to BatchNorm conversion**: Implements the sophisticated training methodology

## Key Modifications

### New Functions Added

- `quad(x)`: Quadratic activation function `0.125*x² + 0.25*x + 0.5`
- `QuadGELU`: GELU replacement using quad function
- `softmax_2QUAD`: Softmax with `(x+5)²` transformation
- `replace_activations_with_quad()`: Replaces all GELU activations in BERT
- `load_pretrained_quad_model()`: Loads pre-trained model weights

### Training Process

1. **Load base BERT model** from Hugging Face
2. **Replace GELU with QuadGELU** throughout the model
3. **Load pre-trained quad weights** (if provided) 
4. **Replace LayerNorm with LinearNorm** for gradual transition
5. **Train with gradual transition** from LayerNorm to BatchNorm
6. **BatchNorm recalibration** phase

## Usage

### Basic Training from Scratch
```bash
python train_quad.py --amp --batch-size 16 --tag quad_from_scratch
```

### Training from Pre-trained Quad Model
```bash
python train_quad.py --amp --batch-size 16 --tag quad_from_pretrained \
       --pretrained-quad-model /path/to/your/pretrained_quad_model.pth
```

### Distributed Training
```bash
torchrun --nproc_per_node=4 train_quad.py --amp --batch-size 8 \
         --pretrained-quad-model /path/to/your/pretrained_quad_model.pth
```

### Using the Helper Script
```bash
python run_quad_training.py --pretrained-model /path/to/model.pth --batch-size 16
```

## Configuration

The training can be configured through `config.py`. Key parameters:

- `MODEL.PRETRAINED_QUAD_PATH`: Path to pre-trained quad model
- `MODEL.TRANSITION_TOTAL_STEPS`: Total steps for LayerNorm→BatchNorm transition (default: 300,000)
- `MODEL.TRANSITION_WARMUP_STEPS`: Warmup steps before transition starts (default: 0)
- `MODEL.TRANSITION_R0`: Transition rate parameter (default: 1.0)
- `TRAIN.EPOCHS_STSB/RTE`: Training epochs (default: 300)
- `TRAIN.RECALIBRATION_EPOCHS`: BatchNorm recalibration epochs (default: 10)

## Pre-trained Model Format

The pre-trained model checkpoint should contain:
- Model state dict with quad activation functions already applied
- Compatible with BERT-base architecture
- Classification head weights will be automatically removed/reinitialized

## Output

Training produces:
- **Best model checkpoints** for each task (STS-B and RTE)
- **Detailed logs** with training progress, scores, and timing
- **Separate directories** for each task (`./results/stsb_gradual/`, `./results/rte_gradual/`)

## Architecture Details

### QuadGELU vs GELU
- **GELU**: `x * Φ(x)` where Φ is cumulative distribution function
- **QuadGELU**: `0.125*x² + 0.25*x + 0.5` (simpler quadratic approximation)

### 2Quad Softmax
- **Standard Softmax**: `exp(x) / Σexp(x)`  
- **2Quad Softmax**: `(x+5)² / Σ(x+5)²` (quadratic transformation)

### LinearNorm Transition
- **Gradual blend**: `λ * LayerNorm(x) + (1-λ) * RepBN(x)`
- **λ decreases** from 1.0 to 0.0 over `TRANSITION_TOTAL_STEPS`
- **RepBN**: BatchNorm + learnable residual connection

## Results

The training evaluates on GLUE benchmarks:
- **STS-B**: Semantic Textual Similarity (Pearson correlation)
- **RTE**: Recognizing Textual Entailment (Accuracy)

Best scores are saved and logged for comparison. 