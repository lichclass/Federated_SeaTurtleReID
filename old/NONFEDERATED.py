import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import math

# --- PyTorch/Torchvision Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast_mode, grad_scaler

import torchvision.transforms as T
from torchvision.models import (
    convnext_base,
    ConvNeXt_Base_Weights,

    # Methods to compare
    swin_b,
    Swin_B_Weights,
    resnet50,
    ResNet50_Weights,
    densenet121,
    DenseNet121_Weights,
)

# --- Library Imports ---
from wildlife_datasets.datasets import SeaTurtleID2022
from wildlife_tools.data import ImageDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from wildlife_datasets.splits import ClosedSetSplit


os.environ['KAGGLE_USERNAME'] = "nashadammuoz"
os.environ['KAGGLE_KEY'] = "KGAT_9f227e36a409b0debe5ee7a27090bd72"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


class DenseNet121Backbone(nn.Module):
    def __init__(self, embedding_dim=512, dropout=0.2):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1
        model = densenet121(weights=weights)

        # Remove original classifier
        in_features = model.classifier.in_features
        model.classifier = nn.Identity()
        self.backbone = model

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.proj(feat)
        # Return normalized embedding and norms (for AdaFace)
        norms = torch.norm(emb, p=2, dim=1, keepdim=True)
        emb = F.normalize(emb, dim=1)
        return emb, norms.squeeze()

class ResNet50Backbone(nn.Module):
    def __init__(self, embedding_dim=512, dropout=0.2):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)

        # Remove original classifier
        in_features = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.proj(feat)
        # Return normalized embedding and norms (for AdaFace)
        norms = torch.norm(emb, p=2, dim=1, keepdim=True)
        emb = F.normalize(emb, dim=1)
        return emb, norms.squeeze()


class SwinTransformerBackbone(nn.Module):
    def __init__(self, embedding_dim=512, dropout=0.2):
        super().__init__()
        weights = Swin_B_Weights.IMAGENET1K_V1
        model = swin_b(weights=weights)

        # Remove original classifier
        in_features = model.head.in_features
        model.head = nn.Identity()
        self.backbone = model

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.proj(feat)
        # Return normalized embedding and norms (for AdaFace)
        norms = torch.norm(emb, p=2, dim=1, keepdim=True)
        emb = F.normalize(emb, dim=1)
        return emb, norms.squeeze()

class ConvNeXtBackbone(nn.Module):
    def __init__(self, embedding_dim=512, dropout=0.2):
        super().__init__()
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
        model = convnext_base(weights=weights)
        # Remove original classifier
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        self.backbone = model
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.proj(feat)
        # Return normalized embedding and norms (for AdaFace)
        norms = torch.norm(emb, p=2, dim=1, keepdim=True)
        emb = F.normalize(emb, dim=1)
        return emb, norms.squeeze()


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        # Precompute trig values
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, norms, label):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        if label is None:
            return cosine * self.s

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class AdaFaceHead(nn.Module):
    def __init__(self, embedding_size, num_classes, m=0.35, h=0.2, s=64., t_alpha=0.01):
        super().__init__()
        self.num_classes = num_classes
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        self.register_buffer('batch_mean', torch.ones(1)*20)
        self.register_buffer('batch_std', torch.ones(1)*100)

    def forward(self, embeddings, norms, label):
        kernel_norm = torch.nn.functional.normalize(self.kernel, dim=0)
        cosine = torch.mm(embeddings, kernel_norm).clamp(-1+1e-3, 1-1e-3)

        if label is None:
            return cosine * self.s

        with torch.no_grad():
            self.batch_mean = norms.mean() * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = norms.std() * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (norms - self.batch_mean) / (self.batch_std + 1e-3)
        margin_scaler = torch.clip(margin_scaler * self.h, -1, 1)

        # AdaFace logic
        m_arc = torch.zeros_like(cosine)
        m_arc.scatter_(1, label.view(-1, 1), 1.0)
        g_angular = -self.m * margin_scaler
        m_arc = m_arc * g_angular.unsqueeze(1)
        
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=1e-3, max=math.pi-1e-3)
        cosine_m = theta_m.cos()

        m_cos = torch.zeros_like(cosine)
        m_cos.scatter_(1, label.view(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add.unsqueeze(1)
        
        return (cosine_m - m_cos) * self.s

# Wrapper Model
class ReIDModel(nn.Module): 
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, labels=None):
        emb, norms = self.backbone(x)
        if labels is not None:
            logits = self.head(emb, norms, labels)
            return logits
        return emb


# --- Utilities ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def extract_features(model, dataset, device, batch_size=16):
    model.eval()
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_features = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting features", leave=False):
            imgs = imgs.to(device)
            features = model(imgs)
            all_features.append(features.cpu().numpy())
    return np.vstack(all_features)

def compute_cosine_similarity(query_features, gallery_features):
    query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
    gallery_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)
    
    similarity_matrix = np.dot(query_norm, gallery_norm.T)
    return similarity_matrix

def evaluate(model, gallery_set, query_set, device, batch_size=16):
    was_training = model.training
    model.eval()

    gallery_features = extract_features(model, gallery_set, device, batch_size)
    query_features = extract_features(model, query_set, device, batch_size)

    similarity_matrix = compute_cosine_similarity(query_features, gallery_features) 

    query_labels = np.array(query_set.labels_string)
    gallery_labels = np.array(gallery_set.labels_string)

    topk_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :5] 

    rank1_acc = 0
    rank5_acc = 0
    
    for i, q_label in enumerate(query_labels):
        retrieved_labels = gallery_labels[topk_indices[i]]
        
        if retrieved_labels[0] == q_label:
            rank1_acc += 1
            
        if q_label in retrieved_labels:
            rank5_acc += 1

    rank1_acc = (rank1_acc / len(query_labels)) * 100.0
    rank5_acc = (rank5_acc / len(query_labels)) * 100.0

    if was_training:
        model.train()
    
    return rank1_acc, rank5_acc


def clean_path(p):
    if 'turtles-data/data/' in p:
        return p.replace('turtles-data/data/', '')
    return p


def data_loader(config):
    print("--- Loading Data ---")
    SeaTurtleID2022.get_data(root=config['root'])

    # 1. Load Base Data
    if config['body_part'] is None:
        dataset_df = SeaTurtleID2022(root=config['root']).df
    else:
        dataset_df = SeaTurtleID2022(root=config['root'], category_name=config['body_part'], img_load='bbox').df
    
    print(f"Original Dataset Size: {len(dataset_df)}")

    # 2. Load Metadata
    try:
        meta_path = Path(config['root']) / 'turtles-data' / 'data' / 'metadata_splits.csv'
        meta_df = pd.read_csv(meta_path)
    except FileNotFoundError:
        raise FileNotFoundError("Metadata splits file not found.")

    # 3. Merge
    dataset_df['join_key'] = dataset_df['path'].apply(clean_path)
    merged_df = pd.merge(
        dataset_df, 
        meta_df[['file_name', 'split_closed', 'split_open']], 
        left_on='join_key', 
        right_on='file_name',
        how='inner' # Changed to inner to avoid NaNs if keys don't match
    )
    print(f"Merged Dataset Size: {len(merged_df)}")

    # 4. Create Base Splits
    split_col = f'split_{config["set"]}'
    train_df = merged_df[merged_df[split_col] == 'train'].reset_index(drop=True)
    valid_df = merged_df[merged_df[split_col] == 'valid'].reset_index(drop=True)
    test_df = merged_df[merged_df[split_col] == 'test'].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(valid_df)}, Test: {len(test_df)}")

    # 5. Helper to safely split Query/Gallery
    def safe_split(df, name):
        if len(df) == 0:
            print(f"WARNING: {name} dataframe is empty!")
            return df, df # Return empty
            
        splitter = ClosedSetSplit(ratio_train=0.5, seed=config['seed'])
        # Pass values directly to avoid index confusion
        splits = splitter.split(df)
        
        if len(splits) == 0:
             print(f"WARNING: Splitter returned no splits for {name}")
             return df, df

        gallery_idx, query_idx = splits[0]
        
        # Verify indices are valid
        if gallery_idx.max() >= len(df) or query_idx.max() >= len(df):
             raise IndexError(f"Splitter returned invalid indices for {name}. Max idx: {gallery_idx.max()}, DF len: {len(df)}")

        gal_df = df.iloc[gallery_idx].reset_index(drop=True)
        qry_df = df.iloc[query_idx].reset_index(drop=True)
        return gal_df, qry_df

    # 6. Apply Split
    val_gallery_df, val_query_df = safe_split(valid_df, "Validation")
    test_gallery_df, test_query_df = safe_split(test_df, "Test")

    # 7. Transforms
    t_train = T.Compose([
        T.Resize((config['image_size'], config['image_size'])),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    t_eval = T.Compose([
        T.Resize((config['image_size'], config['image_size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 8. Create Datasets
    train_set = ImageDataset(train_df, root=config['root'], transform=t_train, col_path='path', col_label='identity')
    val_gallery_set = ImageDataset(val_gallery_df, root=config['root'], transform=t_eval, col_path='path', col_label='identity')
    val_query_set = ImageDataset(val_query_df, root=config['root'], transform=t_eval, col_path='path', col_label='identity')
    test_gallery_set = ImageDataset(test_gallery_df, root=config['root'], transform=t_eval, col_path='path', col_label='identity')
    test_query_set = ImageDataset(test_query_df, root=config['root'], transform=t_eval, col_path='path', col_label='identity')  

    return train_set, (val_gallery_set, val_query_set), (test_gallery_set, test_query_set)


# --- Main ---
def main(config):

    print("Configuration:")
    print("-" * 60)
    for key, value in config.items():
        print(f"{key:15s}: {value}")
    print("-" * 60 + "\n")

    # Create results directory
    results_path = Path(config['results_root']) / config['results_name']
    results_path.mkdir(parents=True, exist_ok=True)

    # Saving configs for this run
    with open(os.path.join(results_path, 'configs.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    set_seed(config['seed'])
    Path(config['root']).mkdir(parents=True, exist_ok=True)

    # Data Loading
    train_set, (val_gallery_set, val_query_set), (test_gallery_set, test_query_set) = data_loader(config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=12, pin_memory=True)
    
    # Build Model
    num_classes = train_set.num_classes

    # Model Components
    if config['backbone'] == 'convnext':
        backbone = ConvNeXtBackbone(embedding_dim=config['embedding_dim'])
    elif config['backbone'] == 'swin':
        backbone = SwinTransformerBackbone(embedding_dim=config['embedding_dim'])
    elif config['backbone'] == 'resnet50':
        backbone = ResNet50Backbone(embedding_dim=config['embedding_dim'])
    elif config['backbone'] == 'densenet121':
        backbone = DenseNet121Backbone(embedding_dim=config['embedding_dim'])

    if config['head'] == 'adaface':
        head = AdaFaceHead(embedding_size=config['embedding_dim'], num_classes=num_classes)
    elif config['head'] == 'arcface':
        head = ArcFaceHead(embedding_size=config['embedding_dim'], num_classes=num_classes)

    model = ReIDModel(backbone=backbone, head=head).to(config['device'])

    if config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['w_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['w_decay'])
    
    criterion = nn.CrossEntropyLoss()

    warmup_epochs = 5
    main_scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'] - warmup_epochs, eta_min=1e-6)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
    
    early_stop_counter = 0
    best_rank1 = 0.0
    best_epoch = -1
    history = {
        'train_loss': [],
        'val_rank1': [],
        'val_rank5': [],
    }

    scaler = grad_scaler.GradScaler()
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        running_loss = 0.
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} - Training", total=len(train_loader))

        for imgs, labels in pbar:
            imgs, labels = imgs.to(config['device']), labels.to(config['device'])
            optimizer.zero_grad()

            with autocast_mode.autocast(device_type='cuda'):
                logits = model(imgs, labels)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()

            if config['optimizer'] == 'sgd':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
        scheduler.step()
        epoch_loss = running_loss / len(train_set)
        print(f"Epoch {epoch} Training Loss: {epoch_loss:.4f}")
        history['train_loss'].append(epoch_loss)

        # Validation Evaluation
        val_rank1, val_rank5 = evaluate(model, val_gallery_set, val_query_set, device=config['device'], batch_size=config['batch_size'])
        model.train()
        history['val_rank1'].append(val_rank1)
        history['val_rank5'].append(val_rank5)
        print(f"Validation Rank-1 Accuracy: {val_rank1:.2f}%, Rank-5 Accuracy: {val_rank5:.2f}%")
        if val_rank1 > best_rank1:
            best_rank1 = val_rank1
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(results_path, 'best_model.pth'))
            print(f"✅ New best model saved at epoch {epoch} with Rank-1 Accuracy: {best_rank1:.2f}%")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= config['patience']:
                print(f"Early stopping triggered after {config['patience']} epochs without improvement.")
                break

        torch.cuda.empty_cache()
    
    print(f"Training completed. Best Validation Rank-1 Accuracy: {best_rank1:.2f}% at epoch {best_epoch}")

    # Saving training history
    with open(os.path.join(results_path, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # Test Evaluation
    model.load_state_dict(torch.load(os.path.join(results_path, 'best_model.pth')))
    model.eval()
    with torch.no_grad():
        test_rank1, test_rank5 = evaluate(model, test_gallery_set, test_query_set, device=config['device'], batch_size=config['batch_size'])
    print(f"Test Rank-1 Accuracy: {test_rank1:.2f}%, Rank-5 Accuracy: {test_rank5:.2f}%")

    # Saving Results
    with open(os.path.join(results_path, 'results.txt'), 'w') as f:
        f.write(f"Best Validation Rank-1 Accuracy: {best_rank1:.2f}% at epoch {best_epoch}\n")
        f.write(f"Test Rank-1 Accuracy: {test_rank1:.2f}%, Rank-5 Accuracy: {test_rank5:.2f}%\n")


# --- Main File ----
if __name__ == "__main__":
    config = {
        'root': './data/SeaTurtleID2022',
        'results_root': './results/non_federated_reid', 
        'description': '',
        'image_size': 224,
        'batch_size': 128,
        'epochs': 100,

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'body_part': 'head',  # 'head', 'turtle', 'flipper', or None for full image
        'set': 'closed',      # 'closed' or 'open'
        'lr': 1e-4,
        'w_decay': 1e-4,

        'backbone': 'convnext',  # 'convnext', 'swin', 'resnet50'
        'embedding_dim': 512,
        'head': 'adaface',      # 'adaface' or 'arcface'

        'optimizer': 'adamw',    # 'adamw' or 'sgd'

        'patience': 10,

    }

    experiments = [
        {
            'results_name': 'RESNET50_ARCFACE',
            'backbone': 'resnet50',
            'head': 'arcface',
            'seeds': [42],
        },
        {
            'results_name': 'SWIN_ARCFACE',
            'backbone': 'swin',
            'head': 'arcface',
            'seeds': [42],
            'optimizer': 'sgd',
            'lr': 1e-3,
            'w_decay': 5e-4,
        },
        {
            'results_name': 'CONVNEXT_ARCFACE',
            'backbone': 'convnext',
            'head': 'arcface',
            'seeds': [42],
        },
        {
            'results_name': 'RESNET50_ADAFACE',
            'backbone': 'resnet50',
            'head': 'adaface',
            'seeds': [42],
        },
        {
            'results_name': 'SWIN_ADAFACE',
            'backbone': 'swin',
            'head': 'adaface',
            'seeds': [42],
            'optimizer': 'sgd',
            'lr': 1e-3,
            'w_decay': 5e-4,
        },
        {
            'results_name': 'CONVNEXT_ADAFACE',
            'backbone': 'convnext',
            'head': 'adaface',
            'seeds': [42],
        },
    ]

    for exp in experiments:
        print(f"Starting {exp['results_name']}...")
        for seed in exp['seeds']:
            exp_config = config.copy()
            exp_config.update(exp)
            exp_config['results_name'] = f"{exp['results_name']}_SEED_{seed}"
            exp_config['seed'] = seed
            exp_config.pop('seeds', None)
            main(exp_config)