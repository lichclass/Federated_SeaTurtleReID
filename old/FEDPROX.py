import json
import os
import math
import warnings
import copy
import numpy as np
import pandas as pd
import time

from pathlib import Path
from tqdm import tqdm

# --- PyTorch/Torchvision Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast_mode, grad_scaler

import torchvision.transforms as T

from torchvision.models import convnext_base, ConvNeXt_Base_Weights

# --- Library Imports ---
from wildlife_datasets.datasets import SeaTurtleID2022
from wildlife_tools.data import ImageDataset
from wildlife_datasets.splits import ClosedSetSplit

os.environ['KAGGLE_USERNAME'] = "nashadammuoz"
os.environ['KAGGLE_KEY'] = "KGAT_9f227e36a409b0debe5ee7a27090bd72"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt Backbone for Feature Extraction.
    """
    def __init__(self, embedding_dim=512, dropout=0.2, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
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
        norms = torch.norm(emb, p=2, dim=1, keepdim=True)
        emb = F.normalize(emb, dim=1)
        return emb, norms.flatten()


class AdaFaceHead(nn.Module):
    """
    AdaFace Implementation https://arxiv.org/abs/2204.00964
    """
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
            std = norms.std() if norms.size(0) > 1 else torch.tensor(0.0, device=norms.device) # Handling for batch size 1
            self.batch_mean = norms.mean() * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

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
    """
    ReID Model combining Backbone and Head.
    """
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, labels=None):
        emb, norms = self.backbone(x)
        if labels is not None:
            logits = self.head(emb, norms, labels)
            return logits, emb
        return emb


def set_seed(seed):
    """
    Set random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def extract_features(model, dataset, device, custom_desc='', batch_size=16):
    """
    Extract features from the dataset using the model.
    """
    model.eval()
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_features = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=custom_desc, leave=False):
            imgs = imgs.to(device)
            features = model(imgs)
            all_features.append(features.cpu().numpy())
    return np.vstack(all_features)


def compute_cosine_similarity(query_features, gallery_features):
    """
    Compute cosine similarity between query and gallery features.
    """
    query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
    gallery_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)
    
    similarity_matrix = np.dot(query_norm, gallery_norm.T)
    return similarity_matrix


def mean_average_precision(similarity_matrix, query_labels, gallery_labels, at=1):
    """
    Compute Mean Average Precision (mAP) at specified rank.
    """
    num_queries = similarity_matrix.shape[0]
    average_precisions = []

    for i in range(num_queries):
        sim_scores = similarity_matrix[i]
        sorted_indices = np.argsort(sim_scores)[::-1]
        sorted_gallery_labels = gallery_labels[sorted_indices]

        relevant_indices = np.where(sorted_gallery_labels == query_labels[i])[0]
        if len(relevant_indices) == 0:
            continue

        hits = 0
        precision_at_k = 0.0
        for rank, idx in enumerate(relevant_indices[:at], start=1):
            hits += 1
            precision_at_k += hits / (idx + 1)

        average_precision = precision_at_k / min(len(relevant_indices), at)
        average_precisions.append(average_precision)

    if len(average_precisions) == 0:
        return 0.0
    return np.mean(average_precisions)


def evaluate(model, gallery_set, query_set, device, batch_size=16, client_id=None, mAP_at=[1,5]):
    """
    Evaluate the model using Rank-1 and Rank-5 accuracy.
    """
    was_training = model.training
    model.eval()

    if client_id is not None:   
        custom_desc = f"Client {client_id} - Extracting gallery features"
    else:
        custom_desc = "Extracting gallery features"
    gallery_features = extract_features(model, gallery_set, device, custom_desc=custom_desc, batch_size=batch_size)

    if client_id is not None:   
        custom_desc = f"Client {client_id} - Extracting query features"
    else:
        custom_desc = "Extracting query features"
    query_features = extract_features(model, query_set, device, custom_desc=custom_desc, batch_size=batch_size)
    
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

    rank1_acc = (rank1_acc / len(query_labels))
    rank5_acc = (rank5_acc / len(query_labels))

    if was_training:
        model.train()

    mAP = []
    for at in mAP_at:
        mAP_score = mean_average_precision(similarity_matrix, query_labels, gallery_labels, at=at)
        mAP.append(mAP_score)
    
    return rank1_acc, rank5_acc, mAP


def clean_path(p):
    if 'turtles-data/data/' in p:
        return p.replace('turtles-data/data/', '')
    return p


def safe_split(df, name, seed):
    if len(df) == 0:
        raise ValueError(f"DataFrame for {name} is empty, cannot split.")
        
    splitter = ClosedSetSplit(ratio_train=0.5, seed=seed)
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


def generate_query_gallery_splits(df, seed, img_size, root):
    gallery_df, query_df = safe_split(df, "Dataset Split", seed)
    t_eval = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    gallery_set = ImageDataset(
        gallery_df,
        root=root,
        transform=t_eval,
        col_path='path',
        col_label='identity'
    )
    query_set = ImageDataset(
        query_df,
        root=root,
        transform=t_eval,
        col_path='path',
        col_label='identity'
    )
    return gallery_set, query_set


def partition_data(df, num_clients, seed, overlap_ratio=0.1, max_client_ratio=0.4):
    all_identities = sorted(df['identity'].unique().tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(all_identities)

    # 1. Split Identities into Public (Shared) and Private
    num_shared = int(len(all_identities) * overlap_ratio)
    shared_identities = all_identities[:num_shared]
    private_identities = all_identities[num_shared:]

    print(f"Total Identities: {len(all_identities)} | Shared: {len(shared_identities)} | Private: {len(private_identities)}")

    # This map tracks which clients get which IDENTITY (Logic Map)
    identity_to_clients_map = {} 
    
    # This map tracks the actual IMAGE INDICES per client (Data Map)
    client_image_indices = {i: [] for i in range(num_clients)}

    # Max clients a shared identity can belong to
    max_clients_limit = max(2, int(num_clients * max_client_ratio))

    # 3. Assign Shared Identities (Multi-client)
    for identity in shared_identities: 
        n_partners = rng.randint(2, max_clients_limit + 1)
        assigned_clients = rng.choice(num_clients, size=n_partners, replace=False)
        identity_to_clients_map[identity] = assigned_clients

    # 4. Assign Private Identities (Single-client)
    for identity in private_identities:
        assigned_client = rng.randint(0, num_clients)
        # Store as a list of 1 so the logic below is consistent
        identity_to_clients_map[identity] = [assigned_client]

    # 5. Pre-shuffle images
    id_to_indices = {id: df[df['identity'] == id].index.tolist() for id in all_identities}
    for id in id_to_indices:
        rng.shuffle(id_to_indices[id])

    # 6. Distribute Images
    for identity in all_identities:
        assigned_clients = identity_to_clients_map[identity]
        all_imgs = id_to_indices[identity]
        
        # 5. Calculate split size
        total_shares = len(assigned_clients)
        imgs_per_client = len(all_imgs) // total_shares
        
        for i, client_id in enumerate(assigned_clients):
            start = i * imgs_per_client
            # If last client, take all remaining to handle odd divisions
            if i == total_shares - 1:
                end = len(all_imgs)
            else:
                end = (i + 1) * imgs_per_client
                
            # Add these specific image rows to the client's pile
            if end > start:
                client_image_indices[client_id].extend(all_imgs[start:end])

    # 7. Build Final DataFrames
    client_dfs = []
    for client_id in range(num_clients):
        indices = client_image_indices[client_id]
        # Use .iloc to fetch rows by integer index
        client_df = df.loc[indices].copy().reset_index(drop=True)
        
        # Optional: Add metadata for debugging
        client_df['is_shared'] = client_df['identity'].isin(shared_identities)
        
        client_dfs.append(client_df)
    
    return client_dfs


def move_to_device(obj, device):
    """
    Recursively move tensors in a nested structure to the device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif hasattr(obj, 'to'):  # Handle models/modules
        return obj.to(device)
    return obj

def optimizer_to(optim, device):
    """
    Moves optimizer state (momentum, variance) to the specified device.
    """
    for state in optim.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)


class FederatedClient:
    def __init__(self, client_id, train_df, config, prototype_backbone=None):
        self.client_id = client_id
        self.train_df = train_df.copy()
        self.config = config
        self.device = config['device']
        self.cpu_device = torch.device('cpu')

        self.unique_identities = sorted(self.train_df['identity'].unique().tolist())
        self.num_local_classes = len(self.unique_identities)

        if prototype_backbone is not None:
            backbone = copy.deepcopy(prototype_backbone).to(self.cpu_device)
        else:
            backbone = ConvNeXtBackbone(embedding_dim=config['embedding_dim'], pretrained=False)

        head = AdaFaceHead(embedding_size=config['embedding_dim'], num_classes=self.num_local_classes)
        self.model = ReIDModel(backbone, head).to(self.cpu_device)
        
        # NOTE: Optimizer is NOT initialized here to avoid stale momentum
        
        t_train = T.Compose([
            T.Resize((self.config['image_size'], self.config['image_size'])),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2, 0.1),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5, scale=(0.02, 0.33)),
        ])
        dataset = ImageDataset(
            self.train_df,
            root=self.config['root'],
            transform=t_train,
            col_path='path',
            col_label='identity'
        )

        self.loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4, 
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

        print(f"Client {self.client_id} - Model initialized (CPU-Resident).")

    def train(self, server_msg):
        self.model.to(self.device)

        # --- FIX 1: Initialize Optimizer HERE to avoid stale momentum ---
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['w_decay']
        )
        scaler = grad_scaler.GradScaler()
        optimizer_to(optimizer, self.device)

        global_weights = server_msg['model_state']
        # --- FIX 2: Ensure global params are float32 and detached ---
        global_params = {
            k: v.to(self.device, dtype=torch.float32) 
            for k, v in global_weights.items()
        }

        self.model.backbone.load_state_dict(global_weights, strict=True)
        self.model.train()

        current_lr = server_msg['current_lr']
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        criterion = nn.CrossEntropyLoss()
        loader = self.loader

        mu = self.config['fedprox_mu']

        for epoch in range(self.config['local_epochs']):
            epoch_loss = 0.0
            epoch_proximal_loss = 0.0
            
            pbar = tqdm(loader, desc=f"Client {self.client_id} Epoch {epoch+1}")
            
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                batch_size = imgs.size(0)

                with autocast_mode.autocast(device_type='cuda'):
                    logits, emb = self.model(imgs, labels)
                    loss_cls = criterion(logits, labels)
                    
                    # --- FIX 3: Efficient Proximal Calculation ---
                    # We exit autocast / force float32 for this calculation to avoid overflow
                    proximal_term = 0.0
                    for name, param in self.model.backbone.named_parameters():
                        if name in global_params:
                            # torch.sum(diff**2) is standard L2 squared. 
                            # We use float32 for stability.
                            diff = param.float() - global_params[name]
                            proximal_term += torch.sum(diff**2)

                    proximal_loss = (mu / 2) * proximal_term
                    
                    # Add back to total loss
                    total_loss = loss_cls + proximal_loss
            
                epoch_proximal_loss += proximal_loss.item() * batch_size
                epoch_loss += total_loss.item() * batch_size

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        
        self.model.to(self.cpu_device)
        torch.cuda.synchronize()
        optimizer_to(optimizer, self.cpu_device)
        torch.cuda.empty_cache()

        dataset_len = len(self.train_df)

        return {
            'client_id': self.client_id,
            'model_state': self.model.backbone.state_dict(),
            'num_samples': dataset_len,
            'loss': epoch_loss / dataset_len,
            'proximal_loss': epoch_proximal_loss / dataset_len,
        }

    def evaluate(self, gallery_set, query_set, mAP_at=[1,5]):
        eval_model = ReIDModel(self.model.backbone, nn.Identity()).to(self.device)
        rank1, rank5, mAP = evaluate(
            eval_model, 
            gallery_set, 
            query_set, 
            self.device, 
            client_id=self.client_id, 
            batch_size=self.config['batch_size'],
            mAP_at=mAP_at
        )
        return rank1, rank5, mAP

class FederatedServer:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        self.current_lr = config['lr']

        self.global_backbone = ConvNeXtBackbone(embedding_dim=config['embedding_dim']).to(self.device)

        self.dummy_optimizer = optim.AdamW(self.global_backbone.parameters(), lr=config['lr'], weight_decay=config['w_decay'])
        warmup_rounds = config['warmup_rounds']
        main_scheduler = CosineAnnealingLR(self.dummy_optimizer, T_max=config['rounds'] - warmup_rounds, eta_min=1e-6)
        warmup_scheduler = LinearLR(self.dummy_optimizer, start_factor=0.1, total_iters=warmup_rounds)
        self.scheduler = SequentialLR(self.dummy_optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_rounds])

    def step_scheduler(self):
        self.scheduler.step()
        self.current_lr = self.scheduler.get_last_lr()[0]
        print(f"[Server] Learning Rate updated to: {self.current_lr:.6f}")

    def aggregate(self, client_msgs, mode='hybrid'):
        print("[Server] Aggregating Weights and Prototypes...")

        total_samples = sum(msg['num_samples'] for msg in client_msgs)
        first_msg = client_msgs[0]
        w_factor = first_msg['num_samples'] / total_samples

        agg_state = {}
        for k, v in first_msg['model_state'].items():
            agg_state[k] = torch.zeros_like(v, device=self.device)
            agg_state[k].add_(v.to(self.device, non_blocking=True), alpha=w_factor)

        for i in range(1, len(client_msgs)):
            msg = client_msgs[i]
            w_factor = msg['num_samples'] / total_samples
            state = msg['model_state']
            for k in agg_state:
                agg_state[k].add_(state[k].to(self.device, non_blocking=True), alpha=w_factor)

        self.global_backbone.load_state_dict(agg_state)

    def evaluate(self, gallery_set, query_set, mAP_at=[1,5]):
        eval_model = ReIDModel(self.global_backbone, nn.Identity()).to(self.device)
        rank1, rank5, mAP = evaluate(
            eval_model, 
            gallery_set, 
            query_set, 
            self.device, 
            batch_size=self.config['batch_size'], 
            mAP_at=mAP_at
        )
        return rank1, rank5, mAP

    def distribute(self):
        comm_msg = {
            'model_state': self.global_backbone.state_dict(),
            'current_lr': self.current_lr
        }
        return comm_msg


def main(config):

    set_seed(config['seed'])
    
    # Timers
    start_time = 0
    end_time = 0
    elapsed_time = 0

    print("[System] Starting Federated ReID Experiment...")
    start_time = time.time()

    print("Configuration:")
    print("-" * 60)
    for key, value in config.items():
        print(f"{key:15s}: {value}")
    print("-" * 60 + "\n")

    results_path = Path(config['results_root']) / config['results_name']
    results_path.mkdir(parents=True, exist_ok=True)

    print("--- Loading Data ---")
    SeaTurtleID2022.get_data(root=config['root'])

    if config['body_part'] is None:
        dataset_df = SeaTurtleID2022(root=config['root']).df
    else:
        dataset_df = SeaTurtleID2022(root=config['root'], category_name=config['body_part'], img_load='bbox').df

    print(f"Original Dataset Size: {len(dataset_df)}")

    try:
        meta_path = Path(config['root']) / 'turtles-data' / 'data' / 'metadata_splits.csv'
        meta_df = pd.read_csv(meta_path)
    except FileNotFoundError:
        print("Metadata not found at standard path.")
        meta_df = None

    if meta_df is None:
        found_metas = list(Path(config['root']).rglob('metadata_splits.csv'))
        if found_metas:
            meta_path = found_metas[0]
            print(f"[System] Found metadata at: {meta_path}")
            meta_df = pd.read_csv(meta_path)
    
    if meta_df is not None:
        print(f"[System] Loading metadata from: {meta_path}")
        dataset_df['join_key'] = dataset_df['path'].apply(clean_path)
        
        merged_df = pd.merge(
            dataset_df, 
            meta_df[['file_name', 'split_closed', 'split_open']], 
            left_on='join_key', 
            right_on='file_name',
            how='inner'
        )
        img_root = 'turtles-data/data/'
        merged_df['path'] = merged_df['join_key'].apply(lambda x: str(Path(img_root) / x))
    else:
        print("[WARNING] Metadata not found! Proceeding with raw dataset (splitting might fail).")
        merged_df = dataset_df

    print(f"Merged Dataset Size: {len(merged_df)}")

    split_col = f'split_{config["set"]}'
    train_df = merged_df[merged_df[split_col] == 'train'].reset_index(drop=True)
    valid_df = merged_df[merged_df[split_col] == 'valid'].reset_index(drop=True)
    test_df = merged_df[merged_df[split_col] == 'test'].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(valid_df)}, Test: {len(test_df)}")

    client_dfs = partition_data(
        train_df,
        num_clients=config['num_clients'],
        seed=config['seed'],
        overlap_ratio=config['overlap_ratio'],
        max_client_ratio=config['max_client_ratio']
    )

    print("\n--- Partition Verification ---")
    for i, df in enumerate(client_dfs):
        n_unique = df['identity'].nunique()
        n_shared = df[df['is_shared'] == True]['identity'].nunique()
        print(f"Client {i}: {len(df)} images | {n_unique} IDs | {n_shared} Shared IDs")

    # Generate Query-Gallery splits for validation and testing
    val_gallery_set, val_query_set = generate_query_gallery_splits(valid_df, seed=config['seed'], img_size=config['image_size'], root=config['root'])
    test_gallery_set, test_query_set = generate_query_gallery_splits(test_df, seed=config['seed'], img_size=config['image_size'], root=config['root'])

    server = FederatedServer(config)
    initial_state = server.distribute()['model_state']
    
    clients = []
    for i in range(config['num_clients']):
        client = FederatedClient(i, client_dfs[i], config, prototype_backbone=server.global_backbone)
        clients.append(client)

    print("[Main] Starting Federated Training...")
    best_val_rank1 = 0.0
    early_stopping_counter = 0
    best_round = 0
    mAPs_at = [1,5]
    history = {
        'global_rank1': [],
        'global_rank5': [],
        'global_mAP': {k: [] for k in mAPs_at},
        'losses': {f'loss_C{i}': [] for i in range(config['num_clients'])},
    }
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[System] Data Preparation Time: {elapsed_time/60:.2f} minutes\n")

    for round_idx in range(1, config['rounds'] + 1):
        print(f"\n=== Communication Round {round_idx} / {config['rounds']} ===")
        server_msg = server.distribute()
        round_start_time = time.time()

        client_results = []
        for client in clients:
            client_start_time = time.time()
            results = client.train(server_msg)
            client_results.append(results)
            client_key = f'loss_C{client.client_id}'
            history['losses'][client_key].append(results['loss'])
            client_end_time = time.time()
            client_elapsed = client_end_time - client_start_time
            print(f"[Client {client.client_id}] Training Loss Total/Proximal: {results['loss']:.4f}/{results['proximal_loss']:.4f}, Time: {client_elapsed/60:.2f} minutes")

        agg_start_time = time.time()
        print("[Server] Aggregating Client Models...")
        server.aggregate(client_results)
        server.step_scheduler()
        agg_end_time = time.time()
        agg_elapsed = agg_end_time - agg_start_time
        print(f"[System] Aggregation Time: {agg_elapsed/60:.2f} minutes")

        # --- Validation ---
        eval_start_time = time.time()
        print("[Server] Evaluating Global Model on Validation Set...")
        global_val_r1, global_val_r5, global_val_map = server.evaluate(val_gallery_set, val_query_set, mAP_at=mAPs_at)
        print(f"  (Global Model) Validation Rank-1: {global_val_r1*100:.2f}%, Rank-5: {global_val_r5*100:.2f}%")
        history['global_rank1'].append(global_val_r1)
        history['global_rank5'].append(global_val_r5)
        for k in mAPs_at:
            history['global_mAP'][k].append(global_val_map[mAPs_at.index(k)])


        if global_val_r1 > best_val_rank1:
            best_val_rank1 = global_val_r1
            best_round = round_idx
            early_stopping_counter = 0

            print("[Server] Saving Best Global Model...")
            torch.save(server.global_backbone.state_dict(), results_path / 'best_backbone.pth')
            
            for client in clients:
                torch.save(client.model.backbone.state_dict(), results_path / f'best_backbone_client{client.client_id}.pth')

            print(f"[Server] ✅ New best model found! Evaluating on Validation Set... at round {round_idx} with Rank-1: {best_val_rank1*100:.2f}%")
        else:
            early_stopping_counter += 1
            print(f"[Server] No improvement. Early Stopping Counter: {early_stopping_counter}/{config['patience']}")
            if early_stopping_counter >= config['patience']:
                print("[Server] Early stopping triggered. Ending training.")
                break
        eval_end_time = time.time()
        eval_elapsed = eval_end_time - eval_start_time
        print(f"[System] Evaluation Time: {eval_elapsed/60:.2f} minutes")
        
        round_end_time = time.time()
        round_elapsed = round_end_time - round_start_time
        print(f"[System] Round {round_idx} Time: {round_elapsed/60:.2f} minutes")

    with open(results_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)

    print(f"\n=== Training Complete. Best Validation Rank-1: {best_val_rank1*100:.2f}% at round {best_round} ===")
    print("[Server] Loading Best Model for Final Evaluation...")

    best_state = torch.load(results_path / 'best_backbone.pth')
    server.global_backbone.load_state_dict(best_state)
    global_test_r1, global_test_r5, global_test_map = server.evaluate(test_gallery_set, test_query_set, mAP_at=[1,5])
    
    clients_test_r1, clients_test_r5 = 0.0, 0.0
    clients_test_map = [0.0] * len(mAPs_at)
    for client in clients:
        best_state = torch.load(results_path / f'best_backbone_client{client.client_id}.pth')
        client.model.backbone.load_state_dict(best_state)
        cr1, cr5, cmap = client.evaluate(test_gallery_set, test_query_set, mAP_at=[1,5])
        clients_test_r1 += cr1
        clients_test_r5 += cr5
        clients_test_map = [x + y for x, y in zip(clients_test_map, cmap)]
    clients_test_r1 /= config['num_clients']
    clients_test_r5 /= config['num_clients']
    clients_test_map = [x / config['num_clients'] for x in clients_test_map]

    print(f"[Server] Final Test Set Performance - (Global) Rank-1: {global_test_r1*100:.2f}%, Rank-5: {global_test_r5*100:.2f}%")
    print(f"[Server] Final Test Set Performance - (Cross-Client) Rank-1: {clients_test_r1*100:.2f}%, Rank-5: {clients_test_r5*100:.2f}%")
    print(f"[Server] Final Test Set mAP - (Global) mAP@1: {global_test_map[0]*100:.2f}%, mAP@5: {global_test_map[1]*100:.2f}%")
    print(f"[Server] Final Test Set mAP - (Cross-Client) mAP@1: {clients_test_map[0]*100:.2f}%, mAP@5: {clients_test_map[1]*100:.2f}%")

    with open(results_path / 'results.txt', 'w') as f:
        f.write(f"Best Validation Rank-1 Accuracy: {best_val_rank1*100:.2f}% at round {best_round}\n")
        f.write(f"(Global) Test Accuracies:")
        f.write(f"     Rank-1 Accuracy: {global_test_r1*100:.2f}%")
        f.write(f"     Rank-5 Accuracy: {global_test_r5*100:.2f}%\n")
        f.write(f"     mAP@1: {global_test_map[0]*100:.2f}%")
        f.write(f"     mAP@5: {global_test_map[1]*100:.2f}%\n")
        f.write(f"(Cross-Client) Test Accuracies:")
        f.write(f"     Rank-1 Accuracy: {clients_test_r1*100:.2f}%")
        f.write(f"     Rank-5 Accuracy: {clients_test_r5*100:.2f}%\n")
        f.write(f"     mAP@1: {clients_test_map[0]*100:.2f}%")
        f.write(f"     mAP@5: {clients_test_map[1]*100:.2f}%\n")

if __name__ == "__main__":
    config = {
        'root': './data/SeaTurtleID2022',
        'results_root': './results/federated_reid',
        'image_size': 384,
        'batch_size': 64,
        'patience': 12,

        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,
        'body_part': 'head',  
        'set': 'closed',      
        'lr': 1e-4,
        'w_decay': 1e-4,

        'embedding_dim': 512,
        
        # Federated Settings
        'num_clients': 5,
        'overlap_ratio': 0.1,
        'max_client_ratio': 0.4,
        'local_epochs': 3,
        'rounds': 50,
        'warmup_rounds': 5,

        'fedprox_mu': 0.01,
    }

    experiments = [
        {
            'results_name': 'FEDPROX_CLOSED_SET_HEAD_MU_0.01',
            'body_part': 'head',
            'set': 'closed',
            'fedprox_mu': 0.01,
            'seeds': [42]
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