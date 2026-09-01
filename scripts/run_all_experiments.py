# ---------------------------------------------------------------------------
# Bounded-Rank Attention — Quantum-inspired Quadratic Attention with
# Fourier-Domain Rank Control in Transformer Architectures
#
# This file is part of the research code associated with:
#   Mallick, M.T., Banerjee, S., Mattar, E.A., Bhattacharya, P., Pal, S.,
#   Kumar, A., Turdiev, J., Saha, H.N., & Chakrabarti, A. (2026).
#   "Quantum-inspired Quadratic Attention with Fourier-Domain Rank Control
#    in Transformer Architectures."  Scientific Reports (in press).
#   DOI: 10.1038/s41598-026-60978-w
#
# Original code by the authors above (AI-Lab group under Prof. Amlan
# Chakrabarti, A.K. Choudhury School of IT, University of Calcutta /
# Artificial Intelligence, IIT Kharagpur).
#
# This repository is maintained by Ravindra Mina (25AI60R02), M.Tech (AI),
# IIT Kharagpur, under Prof. Amlan Chakrabarti (MTP-2 supervisor) for
# ongoing extensions of this work.  See ATTRIBUTION.md.
# ---------------------------------------------------------------------------

# run_all_experiments.py
"""
Main script to run all experiments from the paper.
"""

import torch
import argparse
import os
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Run Bounded-Rank Attention Experiments')
    parser.add_argument('--data_dir', type=str, default='./data/tea_pest',
                        help='Path to Tea Pest dataset')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'tea_pest', 'ablation', 'cifar100', 'imagenet', 'lra'],
                        help='Which experiment to run')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--encoding_dim', type=int, default=32, help='Encoding dimension r')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    results = {}
    
    # Configuration
    config = {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': 5,
        'embed_dim': 768,
        'encoding_dim': args.encoding_dim,
        'depth': 12,
        'num_heads': 12,
        'ff_dim': 3072,
        'dropout': 0.1,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 10,
        'data_dir': args.data_dir
    }
    
    if args.experiment in ['all', 'tea_pest']:
        # Run main Tea Pest experiment
        from main_experiment import run_experiment
        results['tea_pest'] = run_experiment()
    
    if args.experiment in ['all', 'ablation']:
        # Run ablation study
        from ablation_study import run_ablation_study
        from train_evaluate import create_dataloaders
        
        train_loader, val_loader, test_loader = create_dataloaders(
            args.data_dir, args.batch_size, config['img_size']
        )
        
        results['ablation'] = run_ablation_study(
            config, train_loader, val_loader, test_loader, device
        )
    
    if args.experiment in ['all', 'cifar100']:
        # Run CIFAR-100 evaluation
        from cifar100_evaluation import run_cifar100_evaluation
        results['cifar100'] = run_cifar100_evaluation()
    
    if args.experiment in ['all', 'imagenet']:
        # Run ImageNet evaluation
        from imagenet_lra_evaluation import run_imagenet_evaluation
        results['imagenet'] = run_imagenet_evaluation()
    
    if args.experiment in ['all', 'lra']:
        # Run LRA evaluation
        from imagenet_lra_evaluation import run_lra_evaluation
        results['lra'] = run_lra_evaluation()
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(args.output_dir, f'all_results_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nAll results saved to {results_file}")
    
    return results


if __name__ == "__main__":
    main()