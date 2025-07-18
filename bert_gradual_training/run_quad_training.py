#!/usr/bin/env python3
"""
Example script for training BERT with quad/2quad functions and gradual LayerNorm to BatchNorm conversion.

Usage:
    # Train from scratch with quad activations
    python train_quad.py --amp --batch-size 16 --tag quad_from_scratch
    
    # Train starting from a pre-trained quad model
    python train_quad.py --amp --batch-size 16 --tag quad_from_pretrained \
           --pretrained-quad-model /path/to/your/pretrained_quad_model.pth
    
    # Train with distributed setup
    torchrun --nproc_per_node=4 train_quad.py --amp --batch-size 8 \
             --pretrained-quad-model /path/to/your/pretrained_quad_model.pth
"""

import subprocess
import sys
import os

def run_training(pretrained_model_path=None, 
                 batch_size=16, 
                 tag="quad_gradual_training",
                 distributed=False,
                 num_gpus=1):
    """
    Run the quad training with specified parameters.
    
    Args:
        pretrained_model_path: Path to pre-trained model with quad/2quad functions
        batch_size: Batch size per GPU
        tag: Experiment tag for output directory
        distributed: Whether to use distributed training
        num_gpus: Number of GPUs for distributed training
    """
    
    # Base command
    if distributed and num_gpus > 1:
        cmd = [
            "torchrun", 
            f"--nproc_per_node={num_gpus}",
            "train_quad.py"
        ]
    else:
        cmd = ["python", "train_quad.py"]
    
    # Add arguments
    cmd.extend([
        "--amp",
        "--batch-size", str(batch_size),
        "--tag", tag
    ])
    
    # Add pretrained model path if provided
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        cmd.extend(["--pretrained-quad-model", pretrained_model_path])
        print(f"Using pre-trained quad model: {pretrained_model_path}")
    else:
        print("Training from scratch with quad activations")
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the training
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    return result.returncode

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run BERT quad training")
    parser.add_argument("--pretrained-model", type=str, 
                        help="Path to pre-trained model with quad/2quad functions")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size per GPU")
    parser.add_argument("--tag", type=str, default="quad_gradual_training",
                        help="Experiment tag")
    parser.add_argument("--distributed", action="store_true",
                        help="Use distributed training")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs for distributed training")
    
    args = parser.parse_args()
    
    exit_code = run_training(
        pretrained_model_path=args.pretrained_model,
        batch_size=args.batch_size,
        tag=args.tag,
        distributed=args.distributed,
        num_gpus=args.num_gpus
    )
    
    sys.exit(exit_code) 