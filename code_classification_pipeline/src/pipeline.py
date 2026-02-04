"""
Complete Classification Pipeline
Orchestrates language detection and BERT classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path

from .language_detector import LanguageDetector
from .bert_classifier import BERTClassifier

logger = logging.getLogger(__name__)


class CodeClassificationPipeline:
    """
    End-to-end pipeline for code classification.
    
    Pipeline stages:
    1. Language Detection (using Qwen LLM)
    2. BERT Classification (using CodeBERT)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.pipeline_config = config['pipeline']
        
        # Initialize components
        self.language_detector = None
        self.bert_classifier = None
        
        # Check which stages are enabled
        self.stages = {stage['name']: stage for stage in self.pipeline_config['stages']}
        
        logger.info("Code Classification Pipeline initialized")
    
    def initialize_stages(self):
        """Initialize enabled pipeline stages."""
        # Stage 1: Language Detection
        if self.stages.get('language_detection', {}).get('enabled', False):
            logger.info("Initializing Language Detection stage...")
            self.language_detector = LanguageDetector(self.config)
        
        # Stage 2: BERT Classification
        if self.stages.get('bert_classification', {}).get('enabled', False):
            logger.info("Initializing BERT Classification stage...")
            self.bert_classifier = BERTClassifier(self.config)
    
    def run_language_detection(self, codes: List[str]) -> List[str]:
        """
        Run language detection on code snippets.
        
        Args:
            codes: List of code snippets
            
        Returns:
            List of detected languages
        """
        if self.language_detector is None:
            raise ValueError("Language detector not initialized")
        
        logger.info("Running language detection...")
        languages = self.language_detector.detect_batch(codes)
        
        # Log distribution
        distribution = self.language_detector.get_language_distribution(languages)
        logger.info(f"Language distribution: {distribution}")
        
        return languages
    
    def run_bert_classification(
        self, 
        codes: List[str], 
        labels: Optional[List[int]] = None,
        mode: str = "inference"
    ) -> List[int]:
        """
        Run BERT classification.
        
        Args:
            codes: List of code snippets
            labels: Optional labels for training
            mode: 'train' or 'inference'
            
        Returns:
            List of predicted labels (in inference mode)
        """
        if self.bert_classifier is None:
            raise ValueError("BERT classifier not initialized")
        
        if mode == "train":
            if labels is None:
                raise ValueError("Labels required for training mode")
            
            logger.info("Training BERT classifier...")
            # Note: This is simplified. In practice, split train/val here
            # For now, assume data is pre-split
            return None
        
        elif mode == "inference":
            logger.info("Running BERT classification...")
            predictions = self.bert_classifier.predict(
                codes, 
                batch_size=self.config['inference']['batch_size']
            )
            return predictions
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame, output_dir: str):
        """
        Train the pipeline.
        
        Args:
            train_df: Training dataframe with 'code' and 'label' columns
            val_df: Validation dataframe
            output_dir: Directory to save models
        """
        self.initialize_stages()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Stage 1: Language Detection (optional - usually pre-computed)
        if self.stages.get('language_detection', {}).get('enabled', False):
            logger.info("Detecting languages for training data...")
            train_languages = self.run_language_detection(train_df['code'].tolist())
            train_df['detected_language'] = train_languages
            
            logger.info("Detecting languages for validation data...")
            val_languages = self.run_language_detection(val_df['code'].tolist())
            val_df['detected_language'] = val_languages
            
            # Save language detection results
            train_df.to_parquet(output_path / 'train_with_languages.parquet')
            val_df.to_parquet(output_path / 'val_with_languages.parquet')
        
        # Stage 2: BERT Classification Training
        if self.stages.get('bert_classification', {}).get('enabled', False):
            logger.info("Training BERT classifier...")
            
            train_data = {
                'codes': train_df['code'].tolist(),
                'labels': train_df['label'].tolist()
            }
            
            val_data = {
                'codes': val_df['code'].tolist(),
                'labels': val_df['label'].tolist()
            }
            
            checkpoint_path = output_path / 'bert_best.pt'
            self.bert_classifier.train(train_data, val_data, str(checkpoint_path))
            
            logger.info(f"BERT classifier saved to {checkpoint_path}")
        
        logger.info("Training completed")
    
    def predict(
        self, 
        test_df: pd.DataFrame, 
        checkpoint_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Run inference on test data.
        
        Args:
            test_df: Test dataframe with 'code' column
            checkpoint_path: Path to BERT checkpoint (if not already loaded)
            
        Returns:
            DataFrame with predictions
        """
        self.initialize_stages()
        
        results = test_df.copy()
        
        # Stage 1: Language Detection
        if self.stages.get('language_detection', {}).get('enabled', False):
            logger.info("Running language detection on test data...")
            detected_languages = self.run_language_detection(test_df['code'].tolist())
            results['detected_language'] = detected_languages
        
        # Stage 2: BERT Classification
        if self.stages.get('bert_classification', {}).get('enabled', False):
            # Load checkpoint if provided
            if checkpoint_path:
                self.bert_classifier.load_checkpoint(checkpoint_path)
            
            logger.info("Running BERT classification on test data...")
            predictions = self.run_bert_classification(
                test_df['code'].tolist(),
                mode="inference"
            )
            results['predicted_label'] = predictions
        
        return results
    
    def predict_with_ensemble(
        self, 
        test_df: pd.DataFrame,
        checkpoint_paths: List[str]
    ) -> pd.DataFrame:
        """
        Run ensemble prediction using multiple BERT models.
        
        Args:
            test_df: Test dataframe
            checkpoint_paths: List of checkpoint paths
            
        Returns:
            DataFrame with ensemble predictions
        """
        logger.info(f"Running ensemble prediction with {len(checkpoint_paths)} models")
        
        all_predictions = []
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            logger.info(f"Running model {i+1}/{len(checkpoint_paths)}")
            self.bert_classifier.load_checkpoint(checkpoint_path)
            preds = self.run_bert_classification(
                test_df['code'].tolist(),
                mode="inference"
            )
            all_predictions.append(preds)
        
        # Majority voting
        all_predictions = np.array(all_predictions)
        ensemble_predictions = []
        
        for i in range(all_predictions.shape[1]):
            votes = all_predictions[:, i]
            # Get most common prediction
            unique, counts = np.unique(votes, return_counts=True)
            ensemble_predictions.append(unique[np.argmax(counts)])
        
        results = test_df.copy()
        results['predicted_label'] = ensemble_predictions
        
        return results
    
    def evaluate(self, test_df: pd.DataFrame, checkpoint_path: Optional[str] = None) -> Dict:
        """
        Evaluate pipeline on test data with ground truth labels.
        
        Args:
            test_df: Test dataframe with 'code' and 'label' columns
            checkpoint_path: Path to BERT checkpoint
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
        
        # Run prediction
        results = self.predict(test_df, checkpoint_path)
        
        # Compute metrics
        y_true = test_df['label'].values
        y_pred = results['predicted_label'].values
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # Log metrics
        logger.info("Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        logger.info(f"\n{metrics['classification_report']}")
        
        return metrics
