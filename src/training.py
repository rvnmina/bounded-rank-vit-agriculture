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

# train_evaluate.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TeaPestDataset(Dataset):
    """
    Tea Pest dataset for agricultural pest classification.
    """
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Class mapping
        self.classes = ['Aphids', 'Mite', 'Tea_eater_caterpillar', 'Thrip', 'Mosquito_bug']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths and labels
        self.samples = []
        split_dir = os.path.join(root_dir, split)
        
        for class_name in self.classes:
            class_dir = os.path.join(split_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(img_size: int = 224):
    """
    Get data transforms for training and evaluation.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for training, validation, and testing.
    """
    train_transform, test_transform = get_transforms(img_size)
    
    train_dataset = TeaPestDataset(data_dir, 'train', train_transform)
    val_dataset = TeaPestDataset(data_dir, 'val', test_transform)
    test_dataset = TeaPestDataset(data_dir, 'test', test_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{total_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate the model.
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute macro F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100. * correct / total,
        'macro_f1': 100. * macro_f1
    }


def measure_attention_time(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_runs: int = 5
) -> float:
    """
    Measure average attention computation time.
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            break
    
    # Measure time
    times = []
    for run in range(num_runs):
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                _ = model(images)
        
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000 / len(dataloader))  # ms per batch
    
    return np.mean(times)


def measure_peak_memory(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> float:
    """
    Measure peak memory usage during inference.
    """
    model.eval()
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    max_memory = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            _ = model(images)
            
            if torch.cuda.is_available():
                current_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            else:
                current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            max_memory = max(max_memory, current_memory)
    
    return max_memory


def compute_effective_rank(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.99
) -> float:
    """
    Compute effective rank of attention matrices.
    """
    model.eval()
    ranks = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            # Get attention from first layer
            if hasattr(model, 'blocks') and hasattr(model.blocks[0], 'attention'):
                _, attention_weights = model.blocks[0].attention(images, return_attention=True)
                if attention_weights is not None:
                    # Compute singular values
                    U, S, V = torch.svd(attention_weights[0])  # First batch
                    S = S.cpu().numpy()
                    
                    # Effective rank
                    energy = np.cumsum(S**2) / np.sum(S**2)
                    rank = np.searchsorted(energy, threshold) + 1
                    ranks.append(rank)
                    break
    
    return np.mean(ranks) if ranks else 0.0


def run_training_pipeline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 10
) -> Dict:
    """
    Run full training pipeline.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Cosine annealing with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs,
        eta_min=1e-6
    )
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # Warmup
        if epoch <= warmup_epochs:
            warmup_lr = lr * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            scheduler.step()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}/{epochs}: "
              f"Train Acc: {train_metrics['accuracy']:.2f}%, "
              f"Val Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final test evaluation
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    return {
        'model': model,
        'test_metrics': test_metrics,
        'best_val_acc': best_val_acc
    }


def compute_condition_number(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """
    Compute condition number of attention matrix.
    """
    model.eval()
    cond_numbers = []
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            
            if hasattr(model, 'blocks') and hasattr(model.blocks[0], 'attention'):
                _, attention_weights = model.blocks[0].attention(images, return_attention=True)
                if attention_weights is not None:
                    S = torch.svd(attention_weights[0])[1].cpu().numpy()
                    S = S[S > 1e-10]  # Filter near-zero singular values
                    if len(S) > 1:
                        cond = S[0] / S[-1]
                        cond_numbers.append(cond)
                    break
    
    return np.mean(cond_numbers) if cond_numbers else 0.0


def robustness_evaluation(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    noise_levels: List[float] = [0.05, 0.1, 0.15],
    num_runs: int = 5
) -> Dict[str, List[float]]:
    """
    Evaluate robustness under additive Gaussian noise.
    """
    model.eval()
    results = {'clean_accuracy': [], 'perturbed_accuracies': []}
    
    for run in range(num_runs):
        clean_correct = 0
        total = 0
        
        # Clean accuracy
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                clean_correct += predicted.eq(labels).sum().item()
        
        clean_acc = 100. * clean_correct / total
        results['clean_accuracy'].append(clean_acc)
        
        perturbed_accs = []
        for noise_std in noise_levels:
            perturbed_correct = 0
            
            with torch.no_grad():
                for images, labels in dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    # Add Gaussian noise
                    noise = torch.randn_like(images) * noise_std
                    images_noisy = images + noise
                    
                    outputs = model(images_noisy)
                    _, predicted = outputs.max(1)
                    perturbed_correct += predicted.eq(labels).sum().item()
            
            perturbed_acc = 100. * perturbed_correct / total
            perturbed_accs.append(perturbed_acc)
        
        results['perturbed_accuracies'].append(perturbed_accs)
    
    # Compute robustness drops
    clean_acc_mean = np.mean(results['clean_accuracy'])
    perturbed_means = np.mean(results['perturbed_accuracies'], axis=0)
    
    robustness_drops = [
        (clean_acc_mean - acc) / clean_acc_mean * 100
        for acc in perturbed_means
    ]
    
    return {
        'clean_accuracy': clean_acc_mean,
        'perturbed_accuracies': perturbed_means,
        'robustness_drops': robustness_drops,
        'noise_levels': noise_levels
    }