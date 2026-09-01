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

# cifar100_evaluation.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def run_cifar100_evaluation():
    """
    Evaluate the proposed method on CIFAR-100 dataset.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    config = {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': 100,
        'embed_dim': 768,
        'encoding_dim': 32,
        'depth': 12,
        'num_heads': 12,
        'ff_dim': 3072,
        'dropout': 0.1,
        'batch_size': 64,
        'epochs': 100,
        'lr': 3e-4,
        'weight_decay': 1e-4
    }
    
    # Data transforms
    transform_train = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2023, 0.1994, 0.2010])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                           std=[0.2023, 0.1994, 0.2010])
    ])
    
    # Load CIFAR-100
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Split into train/val/test (keeping original test for final evaluation)
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # ============================================================
    # Standard Transformer baseline
    # ============================================================
    print("\n" + "="*60)
    print("CIFAR-100: Standard Transformer")
    print("="*60)
    
    set_seed(42)
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
    
    standard_results = run_training_pipeline(
        standard_model, train_loader, val_loader, test_loader, device,
        epochs=config['epochs'], lr=config['lr'], weight_decay=config['weight_decay']
    )
    
    standard_time = measure_attention_time(standard_model, test_loader, device)
    standard_memory = measure_peak_memory(standard_model, test_loader, device)
    
    # ============================================================
    # Proposed Bounded-Rank Attention
    # ============================================================
    print("\n" + "="*60)
    print("CIFAR-100: Bounded-Rank Attention")
    print("="*60)
    
    set_seed(42)
    proposed_model = BoundedRankVisionTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        encoding_dim=config['encoding_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        seed=42
    ).to(device)
    
    proposed_results = run_training_pipeline(
        proposed_model, train_loader, val_loader, test_loader, device,
        epochs=config['epochs'], lr=config['lr'], weight_decay=config['weight_decay']
    )
    
    proposed_time = measure_attention_time(proposed_model, test_loader, device)
    proposed_memory = measure_peak_memory(proposed_model, test_loader, device)
    
    # ============================================================
    # Results
    # ============================================================
    print("\n" + "="*80)
    print("CIFAR-100 RESULTS")
    print("="*80)
    print(f"{'Model':<30} {'Accuracy':<12} {'Attention Time (ms)':<20} {'Memory (MB)':<15}")
    print("-"*80)
    print(f"{'Standard Transformer':<30} {standard_results['test_metrics']['accuracy']:<12.2f} "
          f"{standard_time:<20.2f} {standard_memory:<15.0f}")
    print(f"{'Proposed Method (r=32)':<30} {proposed_results['test_metrics']['accuracy']:<12.2f} "
          f"{proposed_time:<20.2f} {proposed_memory:<15.0f}")
    
    return {
        'standard': {
            'accuracy': standard_results['test_metrics']['accuracy'],
            'attention_time_ms': standard_time,
            'peak_memory_mb': standard_memory
        },
        'proposed': {
            'accuracy': proposed_results['test_metrics']['accuracy'],
            'attention_time_ms': proposed_time,
            'peak_memory_mb': proposed_memory
        }
    }