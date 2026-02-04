"""
Evaluator Module
Handles model evaluation and metrics computation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Evaluator:
    """Handles model evaluation and metrics visualization."""
    
    def __init__(self, label_mapping: Optional[Dict] = None):
        """
        Initialize evaluator.
        
        Args:
            label_mapping: Dictionary mapping label indices to class names
        """
        self.label_mapping = label_mapping
        self.results = {}
    
    def compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        prefix: str = ""
    ) -> Dict:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            prefix: Prefix for metric names
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            f"{prefix}accuracy": accuracy_score(y_true, y_pred),
            f"{prefix}macro_f1": f1_score(y_true, y_pred, average='macro'),
            f"{prefix}weighted_f1": f1_score(y_true, y_pred, average='weighted'),
            f"{prefix}macro_precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
            f"{prefix}macro_recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        }
        
        # Per-class metrics
        if self.label_mapping:
            class_names = [self.label_mapping.get(i, f"Class_{i}") for i in sorted(set(y_true))]
        else:
            class_names = [f"Class_{i}" for i in sorted(set(y_true))]
        
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, f1 in enumerate(per_class_f1):
            class_name = class_names[i] if i < len(class_names) else f"Class_{i}"
            metrics[f"{prefix}f1_{class_name}"] = f1
        
        self.results.update(metrics)
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in a formatted way."""
        logger.info("=" * 60)
        logger.info("EVALUATION METRICS")
        logger.info("=" * 60)
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
        
        logger.info("=" * 60)
    
    def get_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        output_dict: bool = False
    ):
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_dict: Whether to return dict or string
            
        Returns:
            Classification report
        """
        if self.label_mapping:
            target_names = [self.label_mapping.get(i, f"Class_{i}") for i in sorted(set(y_true))]
        else:
            target_names = None
        
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=target_names,
            output_dict=output_dict,
            zero_division=0
        )
        
        if not output_dict:
            logger.info("\nClassification Report:")
            logger.info(report)
        
        return report
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        output_path: Optional[str] = None,
        figsize: tuple = (10, 8)
    ):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save figure
            figsize: Figure size
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        
        if self.label_mapping:
            labels = [self.label_mapping.get(i, f"Class_{i}") for i in range(len(cm))]
        else:
            labels = [f"Class_{i}" for i in range(len(cm))]
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {output_path}")
        
        plt.close()
    
    def plot_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Optional[str] = None,
        figsize: tuple = (12, 6)
    ):
        """
        Plot per-class F1, precision, and recall scores.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save figure
            figsize: Figure size
        """
        f1_scores = f1_score(y_true, y_pred, average=None, zero_division=0)
        precision_scores = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_scores = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        if self.label_mapping:
            labels = [self.label_mapping.get(i, f"Class_{i}") for i in range(len(f1_scores))]
        else:
            labels = [f"Class_{i}" for i in range(len(f1_scores))]
        
        x = np.arange(len(labels))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        ax.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1 Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Scores')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Per-class metrics plot saved to {output_path}")
        
        plt.close()
    
    def analyze_misclassifications(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: Optional[str] = None,
        max_samples: int = 100
    ) -> pd.DataFrame:
        """
        Analyze and save misclassified samples.
        
        Args:
            df: Original dataframe with code samples
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save misclassifications
            max_samples: Maximum number of samples to save
            
        Returns:
            DataFrame with misclassified samples
        """
        misclassified_mask = y_true != y_pred
        misclassified_indices = np.where(misclassified_mask)[0]
        
        logger.info(f"Found {len(misclassified_indices)} misclassifications")
        
        if len(misclassified_indices) == 0:
            return pd.DataFrame()
        
        # Create misclassification dataframe
        misclassified_df = df.iloc[misclassified_indices].copy()
        misclassified_df['true_label'] = y_true[misclassified_indices]
        misclassified_df['predicted_label'] = y_pred[misclassified_indices]
        
        if self.label_mapping:
            misclassified_df['true_class'] = misclassified_df['true_label'].map(self.label_mapping)
            misclassified_df['predicted_class'] = misclassified_df['predicted_label'].map(self.label_mapping)
        
        # Limit samples
        if len(misclassified_df) > max_samples:
            misclassified_df = misclassified_df.head(max_samples)
        
        if output_path:
            misclassified_df.to_csv(output_path, index=False)
            logger.info(f"Misclassifications saved to {output_path}")
        
        return misclassified_df
    
    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df: Optional[pd.DataFrame] = None,
        output_dir: str = "results",
        prefix: str = ""
    ):
        """
        Generate comprehensive evaluation report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            df: Original dataframe (for misclassification analysis)
            output_dir: Output directory for results
            prefix: Prefix for output files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compute metrics
        metrics = self.compute_metrics(y_true, y_pred, prefix=prefix)
        self.print_metrics(metrics)
        
        # Classification report
        report_dict = self.get_classification_report(y_true, y_pred, output_dict=True)
        
        # Save metrics to JSON
        import json
        with open(output_path / f"{prefix}metrics.json", 'w') as f:
            json.dump({**metrics, 'classification_report': report_dict}, f, indent=2)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(
            y_true, 
            y_pred,
            output_path=str(output_path / f"{prefix}confusion_matrix.png")
        )
        
        # Plot per-class metrics
        self.plot_per_class_metrics(
            y_true,
            y_pred,
            output_path=str(output_path / f"{prefix}per_class_metrics.png")
        )
        
        # Analyze misclassifications
        if df is not None:
            self.analyze_misclassifications(
                df,
                y_true,
                y_pred,
                output_path=str(output_path / f"{prefix}misclassifications.csv")
            )
        
        logger.info(f"Evaluation report saved to {output_dir}")
