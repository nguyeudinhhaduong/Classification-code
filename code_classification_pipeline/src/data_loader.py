"""
Data Loader Module
Handles loading and preprocessing of datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading and preprocessing."""
    
    def __init__(self, config: Dict):
        """
        Initialize DataLoader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_config = config['data']
        
    def load_parquet(self, file_path: str) -> pd.DataFrame:
        """
        Load parquet file.
        
        Args:
            file_path: Path to parquet file
            
        Returns:
            DataFrame
        """
        logger.info(f"Loading data from {file_path}")
        df = pd.read_parquet(file_path, engine='pyarrow')
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def load_train_data(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load training data."""
        train_path = Path(self.data_config['raw_data_path']) / self.data_config['train_file']
        df = self.load_parquet(str(train_path))
        
        if max_rows and len(df) > max_rows:
            logger.info(f"Sampling {max_rows} rows from {len(df)}")
            df = df.sample(max_rows, random_state=self.config['project']['seed']).reset_index(drop=True)
        
        return df
    
    def load_val_data(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load validation data."""
        val_path = Path(self.data_config['raw_data_path']) / self.data_config['val_file']
        df = self.load_parquet(str(val_path))
        
        if max_rows and len(df) > max_rows:
            logger.info(f"Sampling {max_rows} rows from {len(df)}")
            df = df.sample(max_rows, random_state=self.config['project']['seed']).reset_index(drop=True)
        
        return df
    
    def load_test_data(self) -> pd.DataFrame:
        """Load test data."""
        test_path = Path(self.data_config['raw_data_path']) / self.data_config['test_file']
        return self.load_parquet(str(test_path))
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_config = self.config['training']
        
        train_df = self.load_train_data(train_config.get('train_max_rows'))
        val_df = self.load_val_data(train_config.get('val_max_rows'))
        test_df = self.load_test_data()
        
        logger.info(f"Data loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        # Ensure code column is string type
        if 'code' in df.columns:
            df['code'] = df['code'].astype(str)
        
        # Handle missing values
        df = df.dropna(subset=['code'])
        
        # Remove duplicates
        before = len(df)
        df = df.drop_duplicates(subset=['code'])
        after = len(df)
        
        if before != after:
            logger.info(f"Removed {before - after} duplicate samples")
        
        return df.reset_index(drop=True)
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        val_size: float = 0.2, 
        test_size: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test.
        
        Args:
            df: Input dataframe
            val_size: Validation set proportion
            test_size: Test set proportion
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        seed = self.config['project']['seed']
        
        # First split: separate test set
        train_val_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=seed,
            stratify=df['label'] if 'label' in df.columns else None
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=seed,
            stratify=train_val_df['label'] if 'label' in train_val_df.columns else None
        )
        
        logger.info(f"Split data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def get_label_mapping(self, df: pd.DataFrame) -> Dict:
        """
        Get label to language mapping.
        
        Args:
            df: DataFrame with 'label' and optionally 'language' columns
            
        Returns:
            Dictionary mapping labels to language names
        """
        if 'label' not in df.columns:
            return {}
        
        if 'language' in df.columns:
            mapping = df[['label', 'language']].drop_duplicates().set_index('label')['language'].to_dict()
        else:
            # Create generic mapping
            unique_labels = sorted(df['label'].unique())
            mapping = {label: f"Language_{label}" for label in unique_labels}
        
        logger.info(f"Label mapping: {mapping}")
        return mapping
    
    def save_processed_data(self, df: pd.DataFrame, filename: str):
        """
        Save processed data.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        output_path = Path(self.data_config['processed_data_path']) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, engine='pyarrow')
        logger.info(f"Saved processed data to {output_path}")
