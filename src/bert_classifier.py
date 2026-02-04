"""
BERT Classifier Module - Stage 2 of Pipeline
Uses CodeBERT for fine-grained code classification.
"""

import os
import gc
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
import logging

logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    """Dataset for code classification."""
    
    def __init__(self, codes, labels=None):
        self.codes = codes
        self.labels = labels
        self.is_test = labels is None
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        if self.is_test:
            return {"text": str(self.codes[idx])}
        return {"text": str(self.codes[idx]), "labels": int(self.labels[idx])}


class DataCollator:
    """Custom collator for batching."""
    
    def __init__(self, tokenizer, max_len, is_train=False, fp16=True, crop_chars=14000):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        self.fp16 = fp16
        self.crop_chars = crop_chars
    
    def _crop_view(self, text, mode, crop_chars):
        """Crop text for data augmentation."""
        if len(text) <= crop_chars:
            return text
        if mode == "head":
            return text[:crop_chars]
        if mode == "tail":
            return text[-crop_chars:]
        start = random.randint(0, max(0, len(text) - crop_chars))
        return text[start:start + crop_chars]
    
    def __call__(self, batch):
        texts = []
        
        if self.is_train:
            # Data augmentation: random crop
            for item in batch:
                t = item["text"]
                r = random.random()
                if r < 0.45:
                    t = self._crop_view(t, "head", self.crop_chars)
                elif r < 0.90:
                    t = self._crop_view(t, "tail", self.crop_chars)
                else:
                    t = self._crop_view(t, "middle", self.crop_chars)
                texts.append(t)
        else:
            texts = [item["text"] for item in batch]
        
        # Tokenize with padding
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_len,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8 if self.fp16 else None,
        )
        
        if "labels" in batch[0]:
            enc["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
        
        return enc


class FGM:
    """Fast Gradient Method for adversarial training."""
    
    def __init__(self, model):
        self.model = model
        self.backup = {}
    
    def attack(self, eps=1.0, emb_name="word_embeddings"):
        """Apply adversarial perturbation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(eps * param.grad / norm)
    
    def restore(self, emb_name="word_embeddings"):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class BERTClassifier:
    """
    CodeBERT-based classifier for code classification.
    """
    
    def __init__(self, config):
        """
        Initialize BERT Classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_name = config['models']['bert_classifier']['model_name']
        self.max_length = config['models']['bert_classifier']['max_length']
        self.num_labels = config['models']['bert_classifier']['num_labels']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        self.best_f1 = -1
        
        logger.info(f"BERT Classifier initialized with {self.n_gpu} GPU(s)")
    
    def load_model(self):
        """Load CodeBERT model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=self.num_labels
        )
        
        if self.n_gpu > 1:
            self.model = nn.DataParallel(self.model)
        
        self.model.to(self.device)
        
        logger.info("Model loaded successfully")
    
    def _unwrap_model(self):
        """Unwrap model from DataParallel."""
        return self.model.module if hasattr(self.model, "module") else self.model
    
    def prepare_optimizer(self, total_steps):
        """Prepare optimizer and scheduler."""
        train_config = self.config['training']
        
        self.optimizer = torch.optim.AdamW(
            self._unwrap_model().parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        warmup_steps = int(total_steps * train_config['warmup_ratio'])
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        self.scaler = torch.amp.GradScaler("cuda", enabled=train_config['fp16'])
    
    def train(self, train_data, val_data, save_path):
        """
        Train the BERT classifier.
        
        Args:
            train_data: Dictionary with 'codes' and 'labels'
            val_data: Dictionary with 'codes' and 'labels'
            save_path: Path to save best model
        """
        if self.model is None:
            self.load_model()
        
        train_config = self.config['training']
        
        # Prepare datasets
        train_dataset = CodeDataset(train_data['codes'], train_data['labels'])
        val_dataset = CodeDataset(val_data['codes'], val_data['labels'])
        
        # Compute class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_data['labels']),
            y=train_data['labels']
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float, device=self.device)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=train_config['shuffle_train'],
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=DataCollator(
                self.tokenizer, 
                self.max_length, 
                is_train=True, 
                fp16=train_config['fp16']
            )
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'] * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=DataCollator(
                self.tokenizer, 
                self.max_length, 
                is_train=False, 
                fp16=train_config['fp16']
            )
        )
        
        # Prepare optimizer
        total_steps = (len(train_loader) // train_config['gradient_accumulation_steps']) * train_config['epochs']
        self.prepare_optimizer(total_steps)
        
        # Optional FGM
        fgm = FGM(self.model) if train_config.get('use_fgm', False) else None
        
        # Training loop
        start_time = time.time()
        time_budget = train_config['time_budget_hours'] * 3600
        safe_margin = train_config['safe_margin_min'] * 60
        
        global_step = 0
        
        logger.info("Starting training...")
        
        for epoch in range(train_config['epochs']):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_config['epochs']}")
            self.optimizer.zero_grad(set_to_none=True)
            
            for step, batch in enumerate(pbar):
                # Time guard
                if time.time() - start_time > (time_budget - safe_margin):
                    logger.warning("Near time limit, stopping training early")
                    break
                
                labels = batch["labels"].to(self.device, non_blocking=True)
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "labels"}
                
                # Forward pass
                with torch.autocast(device_type="cuda", enabled=train_config['fp16']):
                    logits = self.model(**batch).logits
                    loss = F.cross_entropy(
                        logits, 
                        labels,
                        weight=class_weights,
                        label_smoothing=train_config['label_smoothing']
                    ) / train_config['gradient_accumulation_steps']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # FGM adversarial training
                if fgm and global_step / total_steps >= train_config.get('fgm_start_frac', 0.85):
                    fgm.attack(eps=train_config.get('fgm_eps', 0.6))
                    with torch.autocast(device_type="cuda", enabled=train_config['fp16']):
                        logits_adv = self.model(**batch).logits
                        loss_adv = F.cross_entropy(
                            logits_adv, 
                            labels,
                            weight=class_weights,
                            label_smoothing=train_config['label_smoothing']
                        ) / train_config['gradient_accumulation_steps']
                    self.scaler.scale(loss_adv).backward()
                    fgm.restore()
                
                # Optimizer step
                if (step + 1) % train_config['gradient_accumulation_steps'] == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self._unwrap_model().parameters(), 
                        train_config['max_grad_norm']
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    global_step += 1
                
                pbar.set_postfix(loss=float(loss.item() * train_config['gradient_accumulation_steps']))
            
            # Validation
            val_f1, val_acc = self.evaluate(val_loader)
            logger.info(f"Epoch {epoch+1}: Val F1={val_f1:.4f}, Val Acc={val_acc:.4f}")
            
            # Save best model
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                torch.save(self._unwrap_model().state_dict(), save_path)
                logger.info(f"Saved best model with F1={val_f1:.4f}")
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
            
            # Time guard
            if time.time() - start_time > (time_budget - safe_margin):
                break
        
        logger.info(f"Training completed. Best F1: {self.best_f1:.4f}")
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate model on validation set."""
        self.model.eval()
        preds, labels = [], []
        
        for batch in dataloader:
            y = batch["labels"].to(self.device, non_blocking=True)
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k != "labels"}
            
            with torch.autocast(device_type="cuda", enabled=self.config['training']['fp16']):
                out = self.model(**batch).logits
            
            preds.extend(out.argmax(1).detach().cpu().numpy().tolist())
            labels.extend(y.detach().cpu().numpy().tolist())
        
        f1 = f1_score(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        
        return f1, acc
    
    @torch.no_grad()
    def predict(self, codes, batch_size=32):
        """
        Predict labels for code snippets.
        
        Args:
            codes: List of code snippets
            batch_size: Batch size for inference
            
        Returns:
            List of predicted labels
        """
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        
        dataset = CodeDataset(codes)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=DataCollator(
                self.tokenizer, 
                self.max_length, 
                is_train=False, 
                fp16=self.config['training']['fp16']
            )
        )
        
        predictions = []
        
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            with torch.autocast(device_type="cuda", enabled=self.config['training']['fp16']):
                logits = self.model(**batch).logits
            
            predictions.extend(logits.argmax(1).detach().cpu().numpy().tolist())
        
        return predictions
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint."""
        if self.model is None:
            self.load_model()
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self._unwrap_model().load_state_dict(state_dict)
        logger.info("Checkpoint loaded successfully")
