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

# ablation_study.py
import torch
import torch.nn as nn
from collections import OrderedDict

def run_ablation_study(config, train_loader, val_loader, test_loader, device):
    """
    Run component ablation analysis.
    """
    ablation_results = {}
    
    # ============================================================
    # Configuration 1: Standard Dot-Product Attention (softmax)
    # ============================================================
    print("\n" + "="*60)
    print("Ablation: Standard Dot-Product Attention (softmax)")
    print("="*60)
    
    set_seed(42)
    model = StandardVisionTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout']
    ).to(device)
    
    results = run_training_pipeline(
        model, train_loader, val_loader, test_loader, device,
        epochs=config['epochs'], lr=config['lr']
    )
    
    ablation_results['standard_dot_product'] = {
        'accuracy': results['test_metrics']['accuracy'],
        'macro_f1': results['test_metrics']['macro_f1'],
        'attention_time_ms': measure_attention_time(model, test_loader, device),
        'peak_memory_mb': measure_peak_memory(model, test_loader, device),
        'effective_rank': compute_effective_rank(model, test_loader, device)
    }
    
    # ============================================================
    # Configuration 2: Standard Linear Attention
    # ============================================================
    print("\n" + "="*60)
    print("Ablation: Standard Linear Attention")
    print("="*60)
    
    class LinearAttention(nn.Module):
        def __init__(self, embed_dim, dropout=0.1):
            super().__init__()
            self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
            self.dropout = nn.Dropout(dropout)
            
        def forward(self, x):
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)
            
            # Linear attention: (K^T V) then Q
            KV = torch.einsum('bnr,bnd->brd', K, V)
            Z = torch.einsum('bnr,brd->bnd', Q, KV)
            
            return self.W_o(Z)
    
    class LinearTransformer(nn.Module):
        def __init__(self, img_size=224, patch_size=16, num_classes=5, 
                     embed_dim=768, depth=12, ff_dim=3072, dropout=0.1):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.pos_embed = nn.Parameter(torch.randn(1, (img_size//patch_size)**2 + 1, embed_dim) * 0.02)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.pos_drop = nn.Dropout(dropout)
            
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    'attn': LinearAttention(embed_dim, dropout),
                    'ffn': nn.Sequential(
                        nn.Linear(embed_dim, ff_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(ff_dim, embed_dim),
                        nn.Dropout(dropout)
                    ),
                    'norm1': nn.LayerNorm(embed_dim),
                    'norm2': nn.LayerNorm(embed_dim)
                })
                for _ in range(depth)
            ])
            
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)
            
        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            
            for block in self.blocks:
                attn_out = block['attn'](block['norm1'](x))
                x = x + attn_out
                ffn_out = block['ffn'](block['norm2'](x))
                x = x + ffn_out
            
            x = self.norm(x)
            return self.head(x[:, 0])
    
    set_seed(42)
    model = LinearTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout']
    ).to(device)
    
    results = run_training_pipeline(
        model, train_loader, val_loader, test_loader, device,
        epochs=config['epochs'], lr=config['lr']
    )
    
    ablation_results['linear_attention'] = {
        'accuracy': results['test_metrics']['accuracy'],
        'macro_f1': results['test_metrics']['macro_f1'],
        'attention_time_ms': measure_attention_time(model, test_loader, device),
        'peak_memory_mb': measure_peak_memory(model, test_loader, device),
        'effective_rank': compute_effective_rank(model, test_loader, device)
    }
    
    # ============================================================
    # Configuration 3: Standard Quadratic Attention
    # ============================================================
    print("\n" + "="*60)
    print("Ablation: Standard Quadratic Attention")
    print("="*60)
    
    class StandardQuadraticAttention(nn.Module):
        def __init__(self, embed_dim, dropout=0.1):
            super().__init__()
            self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
            self.dropout = nn.Dropout(dropout)
            self.scale = 1.0 / (embed_dim ** 0.25)
            
        def forward(self, x):
            Q = self.W_q(x)
            K = self.W_k(x)
            V = self.W_v(x)
            
            # Quadratic attention: (Q @ K^T) @ V with softmax
            A = torch.einsum('bnd,bmd->bnm', Q, K) * self.scale
            A = F.softmax(A, dim=-1)
            Z = torch.einsum('bnm,bmd->bnd', A, V)
            
            return self.W_o(Z)
    
    class QuadraticTransformer(nn.Module):
        def __init__(self, img_size=224, patch_size=16, num_classes=5,
                     embed_dim=768, depth=12, ff_dim=3072, dropout=0.1):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.pos_embed = nn.Parameter(torch.randn(1, (img_size//patch_size)**2 + 1, embed_dim) * 0.02)
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.pos_drop = nn.Dropout(dropout)
            
            self.blocks = nn.ModuleList([
                nn.ModuleDict({
                    'attn': StandardQuadraticAttention(embed_dim, dropout),
                    'ffn': nn.Sequential(
                        nn.Linear(embed_dim, ff_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(ff_dim, embed_dim),
                        nn.Dropout(dropout)
                    ),
                    'norm1': nn.LayerNorm(embed_dim),
                    'norm2': nn.LayerNorm(embed_dim)
                })
                for _ in range(depth)
            ])
            
            self.norm = nn.LayerNorm(embed_dim)
            self.head = nn.Linear(embed_dim, num_classes)
            
        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            x = x + self.pos_embed
            x = self.pos_drop(x)
            
            for block in self.blocks:
                attn_out = block['attn'](block['norm1'](x))
                x = x + attn_out
                ffn_out = block['ffn'](block['norm2'](x))
                x = x + ffn_out
            
            x = self.norm(x)
            return self.head(x[:, 0])
    
    set_seed(42)
    model = QuadraticTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout']
    ).to(device)
    
    results = run_training_pipeline(
        model, train_loader, val_loader, test_loader, device,
        epochs=config['epochs'], lr=config['lr']
    )
    
    ablation_results['quadratic_attention'] = {
        'accuracy': results['test_metrics']['accuracy'],
        'macro_f1': results['test_metrics']['macro_f1'],
        'attention_time_ms': measure_attention_time(model, test_loader, device),
        'peak_memory_mb': measure_peak_memory(model, test_loader, device),
        'effective_rank': compute_effective_rank(model, test_loader, device)
    }
    
    # ============================================================
    # Configuration 4: Quadratic + Fixed-Dimensional Encoding
    # ============================================================
    print("\n" + "="*60)
    print("Ablation: Quadratic + Fixed-Dimensional Encoding")
    print("="*60)
    
    # This is the BoundedRankVisionTransformer without Fourier
    set_seed(42)
    model = BoundedRankVisionTransformer(
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
    
    results = run_training_pipeline(
        model, train_loader, val_loader, test_loader, device,
        epochs=config['epochs'], lr=config['lr']
    )
    
    ablation_results['quadratic_fixed_encoding'] = {
        'accuracy': results['test_metrics']['accuracy'],
        'macro_f1': results['test_metrics']['macro_f1'],
        'attention_time_ms': measure_attention_time(model, test_loader, device),
        'peak_memory_mb': measure_peak_memory(model, test_loader, device),
        'effective_rank': compute_effective_rank(model, test_loader, device)
    }
    
    # ============================================================
    # Configuration 5: Full Method (with Fourier)
    # ============================================================
    print("\n" + "="*60)
    print("Ablation: Full Method (with Fourier)")
    print("="*60)
    
    # This is the same as the Fourier model from above
    class FourierVisionTransformer(BoundedRankVisionTransformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            from fourier_attention import FourierBoundedRankAttention
            
            class FourierBlock(nn.Module):
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
            
            self.blocks = nn.ModuleList([
                FourierBlock(
                    embed_dim=self.embed_dim,
                    encoding_dim=self.encoding_dim,
                    ff_dim=kwargs.get('ff_dim', 3072),
                    dropout=kwargs.get('dropout', 0.1),
                    seed=kwargs.get('seed', 42) + i
                )
                for i in range(kwargs.get('depth', 12))
            ])
    
    set_seed(42)
    model = FourierVisionTransformer(
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
    
    results = run_training_pipeline(
        model, train_loader, val_loader, test_loader, device,
        epochs=config['epochs'], lr=config['lr']
    )
    
    ablation_results['full_method'] = {
        'accuracy': results['test_metrics']['accuracy'],
        'macro_f1': results['test_metrics']['macro_f1'],
        'attention_time_ms': measure_attention_time(model, test_loader, device),
        'peak_memory_mb': measure_peak_memory(model, test_loader, device),
        'effective_rank': compute_effective_rank(model, test_loader, device)
    }
    
    # ============================================================
    # Print Ablation Summary
    # ============================================================
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Configuration':<40} {'Accuracy':<12} {'Macro-F1':<12} {'Time(ms)':<12} {'Memory(MB)':<12} {'Rank':<10}")
    print("-"*80)
    
    for name, metrics in ablation_results.items():
        print(f"{name:<40} {metrics['accuracy']:<12.2f} {metrics['macro_f1']:<12.2f} "
              f"{metrics['attention_time_ms']:<12.2f} {metrics['peak_memory_mb']:<12.0f} "
              f"{metrics['effective_rank']:<10.0f}")
    
    return ablation_results