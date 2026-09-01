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

# main_experiment.py
import torch
import numpy as np
import random
from datetime import datetime
import json
import os

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_experiment():
    """Run the full experiment pipeline."""
    
    # Configuration
    config = {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': 5,
        'embed_dim': 768,
        'encoding_dim': 32,
        'depth': 12,
        'num_heads': 12,
        'ff_dim': 3072,
        'dropout': 0.1,
        'batch_size': 32,
        'epochs': 100,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 10,
        'seeds': [42, 123, 456, 789, 101112],
        'data_dir': './data/tea_pest'  # Update with your data path
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size']
    )
    
    results = {
        'standard_transformer': {},
        'bounded_rank_attention': {},
        'fourier_attention': {}
    }
    
    # ============================================================
    # Experiment 1: Standard Transformer Baseline
    # ============================================================
    print("\n" + "="*60)
    print("Running Standard Transformer Baseline")
    print("="*60)
    
    set_seed(config['seeds'][0])
    
    standard_model = StandardVisionTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout']
    ).to(device)
    
    # Train standard model
    standard_results = run_training_pipeline(
        standard_model,
        train_loader,
        val_loader,
        test_loader,
        device,
        epochs=config['epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs']
    )
    
    # Measure efficiency
    standard_time = measure_attention_time(standard_model, test_loader, device)
    standard_memory = measure_peak_memory(standard_model, test_loader, device)
    standard_rank = compute_effective_rank(standard_model, test_loader, device)
    standard_cond = compute_condition_number(standard_model, test_loader, device)
    
    # Robustness
    standard_robustness = robustness_evaluation(standard_model, test_loader, device)
    
    results['standard_transformer'] = {
        'accuracy': standard_results['test_metrics']['accuracy'],
        'macro_f1': standard_results['test_metrics']['macro_f1'],
        'attention_time_ms': standard_time,
        'peak_memory_mb': standard_memory,
        'effective_rank': standard_rank,
        'condition_number': standard_cond,
        'robustness_drops': standard_robustness['robustness_drops'],
        'clean_accuracy': standard_robustness['clean_accuracy']
    }
    
    print(f"Standard Transformer - Acc: {results['standard_transformer']['accuracy']:.2f}%, "
          f"Time: {results['standard_transformer']['attention_time_ms']:.2f}ms, "
          f"Memory: {results['standard_transformer']['peak_memory_mb']:.0f}MB")
    
    # ============================================================
    # Experiment 2: Bounded-Rank Quadratic Attention
    # ============================================================
    print("\n" + "="*60)
    print("Running Bounded-Rank Quadratic Attention")
    print("="*60)
    
    set_seed(config['seeds'][0])
    
    bounded_model = BoundedRankVisionTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        encoding_dim=config['encoding_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        seed=config['seeds'][0]
    ).to(device)
    
    # Train bounded model
    bounded_results = run_training_pipeline(
        bounded_model,
        train_loader,
        val_loader,
        test_loader,
        device,
        epochs=config['epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs']
    )
    
    # Measure efficiency
    bounded_time = measure_attention_time(bounded_model, test_loader, device)
    bounded_memory = measure_peak_memory(bounded_model, test_loader, device)
    bounded_rank = compute_effective_rank(bounded_model, test_loader, device)
    bounded_cond = compute_condition_number(bounded_model, test_loader, device)
    
    # Robustness
    bounded_robustness = robustness_evaluation(bounded_model, test_loader, device)
    
    results['bounded_rank_attention'] = {
        'accuracy': bounded_results['test_metrics']['accuracy'],
        'macro_f1': bounded_results['test_metrics']['macro_f1'],
        'attention_time_ms': bounded_time,
        'peak_memory_mb': bounded_memory,
        'effective_rank': bounded_rank,
        'condition_number': bounded_cond,
        'robustness_drops': bounded_robustness['robustness_drops'],
        'clean_accuracy': bounded_robustness['clean_accuracy']
    }
    
    print(f"Bounded-Rank - Acc: {results['bounded_rank_attention']['accuracy']:.2f}%, "
          f"Time: {results['bounded_rank_attention']['attention_time_ms']:.2f}ms, "
          f"Memory: {results['bounded_rank_attention']['peak_memory_mb']:.0f}MB")
    
    # ============================================================
    # Experiment 3: Fourier-Domain Bounded-Rank Attention
    # ============================================================
    print("\n" + "="*60)
    print("Running Fourier-Domain Bounded-Rank Attention")
    print("="*60)
    
    # Create Fourier model
    class FourierVisionTransformer(BoundedRankVisionTransformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Replace attention blocks with Fourier version
            self.blocks = nn.ModuleList([
                TransformerBlockFourier(
                    embed_dim=self.embed_dim,
                    encoding_dim=self.encoding_dim,
                    ff_dim=kwargs.get('ff_dim', 3072),
                    dropout=kwargs.get('dropout', 0.1),
                    seed=kwargs.get('seed', 42) + i
                )
                for i in range(kwargs.get('depth', 12))
            ])
    
    class TransformerBlockFourier(nn.Module):
        def __init__(self, embed_dim, encoding_dim, ff_dim, dropout, seed):
            super().__init__()
            self.attention = FourierBoundedRankAttention(
                embed_dim=embed_dim,
                encoding_dim=encoding_dim,
                dropout=dropout,
                seed=seed
            )
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, embed_dim),
                nn.Dropout(dropout)
            )
            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            
        def forward(self, x):
            attn_out, _ = self.attention(self.norm1(x))
            x = x + attn_out
            ffn_out = self.ffn(self.norm2(x))
            x = x + ffn_out
            return x
    
    set_seed(config['seeds'][0])
    
    fourier_model = FourierVisionTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        encoding_dim=config['encoding_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        seed=config['seeds'][0]
    ).to(device)
    
    # Train Fourier model
    fourier_results = run_training_pipeline(
        fourier_model,
        train_loader,
        val_loader,
        test_loader,
        device,
        epochs=config['epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        warmup_epochs=config['warmup_epochs']
    )
    
    # Measure efficiency
    fourier_time = measure_attention_time(fourier_model, test_loader, device)
    fourier_memory = measure_peak_memory(fourier_model, test_loader, device)
    fourier_rank = compute_effective_rank(fourier_model, test_loader, device)
    fourier_cond = compute_condition_number(fourier_model, test_loader, device)
    
    # Robustness
    fourier_robustness = robustness_evaluation(fourier_model, test_loader, device)
    
    results['fourier_attention'] = {
        'accuracy': fourier_results['test_metrics']['accuracy'],
        'macro_f1': fourier_results['test_metrics']['macro_f1'],
        'attention_time_ms': fourier_time,
        'peak_memory_mb': fourier_memory,
        'effective_rank': fourier_rank,
        'condition_number': fourier_cond,
        'robustness_drops': fourier_robustness['robustness_drops'],
        'clean_accuracy': fourier_robustness['clean_accuracy']
    }
    
    print(f"Fourier - Acc: {results['fourier_attention']['accuracy']:.2f}%, "
          f"Time: {results['fourier_attention']['attention_time_ms']:.2f}ms, "
          f"Memory: {results['fourier_attention']['peak_memory_mb']:.0f}MB")
    
    # ============================================================
    # Save Results
    # ============================================================
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'experiment_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':<12} {'Macro-F1':<12} {'Time(ms)':<12} {'Memory(MB)':<12} {'Rank':<10}")
    print("-"*80)
    
    for name, metrics in results.items():
        print(f"{name:<30} {metrics['accuracy']:<12.2f} {metrics['macro_f1']:<12.2f} "
              f"{metrics['attention_time_ms']:<12.2f} {metrics['peak_memory_mb']:<12.0f} "
              f"{metrics['effective_rank']:<10.0f}")
    
    return results


if __name__ == "__main__":
    results = run_experiment()