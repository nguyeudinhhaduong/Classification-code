"""
Utility Functions
Common helper functions used across the pipeline.
"""

import os
import random
import numpy as np
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import json


def setup_logging(log_dir: str = "logs", log_file: str = "pipeline.log", level: str = "INFO"):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_file: Log filename
        level: Logging level
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path / log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger = logging.getLogger(__name__)
    logger.info(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_json(data: Any, output_path: str):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(file_path: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        file_path: Input file path
        
    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_device_info() -> Dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count(),
        'cuda_device_names': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info['cuda_device_names'].append(torch.cuda.get_device_name(i))
    
    return info


def print_device_info():
    """Print device information."""
    info = get_device_info()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("DEVICE INFORMATION")
    logger.info("=" * 50)
    logger.info(f"CUDA Available: {info['cuda_available']}")
    logger.info(f"GPU Count: {info['cuda_device_count']}")
    
    if info['cuda_device_names']:
        for i, name in enumerate(info['cuda_device_names']):
            logger.info(f"GPU {i}: {name}")
    
    logger.info("=" * 50)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_submission_file(
    ids: np.ndarray, 
    predictions: np.ndarray, 
    output_path: str,
    id_column: str = "id",
    label_column: str = "label"
):
    """
    Create submission CSV file.
    
    Args:
        ids: Sample IDs
        predictions: Predicted labels
        output_path: Output file path
        id_column: Name of ID column
        label_column: Name of label column
    """
    import pandas as pd
    
    submission = pd.DataFrame({
        id_column: ids,
        label_column: predictions
    })
    
    submission.to_csv(output_path, index=False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Submission file saved to {output_path}")
    logger.info(f"Total predictions: {len(submission)}")


def get_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved
        }
    return None


def cleanup_memory():
    """Clean up GPU memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
