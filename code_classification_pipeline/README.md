# Code Classification Pipeline

A comprehensive AI research pipeline for code language detection and classification using Large Language Models (LLM) and BERT-based models.

## 📋 Overview

This pipeline implements a two-stage approach for code classification:

1. **Stage 1: Language Detection** - Uses Qwen LLM to detect programming languages
2. **Stage 2: Fine-grained Classification** - Uses CodeBERT for detailed classification

## 🏗️ Project Structure

```
code_classification_pipeline/
├── configs/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed data files
├── models/
│   ├── pretrained/              # Pre-trained model weights
│   └── checkpoints/             # Training checkpoints
├── src/
│   ├── __init__.py
│   ├── language_detector.py    # LLM-based language detection
│   ├── bert_classifier.py      # BERT classification module
│   ├── pipeline.py              # Main pipeline orchestrator
│   ├── data_loader.py           # Data loading utilities
│   ├── utils.py                 # Helper functions
│   └── evaluator.py             # Evaluation and metrics
├── notebooks/
│   └── exploration.ipynb        # Data exploration notebooks
├── results/                     # Evaluation results and plots
├── logs/                        # Training logs
├── main.py                      # Main training script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to the project directory
cd code_classification_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Place your data files in the `data/raw/` directory:

- `train.parquet` - Training data
- `val.parquet` - Validation data
- `test.parquet` - Test data

Expected data format:

```python
{
    'code': str,      # Code snippet
    'label': int,     # Label (for train/val)
    'id': int         # Sample ID (optional)
}
```

### Configuration

Edit `configs/config.yaml` to customize:

- Model paths and hyperparameters
- Training settings
- Pipeline stages
- Data paths

### Training

```bash
# Train the complete pipeline
python main.py --mode train --output_dir models/checkpoints

# Train with custom config
python main.py --mode train --config configs/custom_config.yaml
```

### Evaluation

```bash
# Evaluate on test set
python main.py --mode eval --checkpoint models/checkpoints/bert_best.pt
```

### Prediction

```bash
# Generate predictions
python main.py --mode predict --checkpoint models/checkpoints/bert_best.pt
```

## 📊 Pipeline Stages

### Stage 1: Language Detection (Qwen LLM)

Uses Qwen-2.5-Coder-1.5B-Instruct to detect programming languages:

- Supports: JavaScript, PHP, Java, Python, C#, C++, Go, C
- Zero-shot classification with carefully crafted prompts
- Distinguishes between similar languages (C vs C++, Java vs C#)

### Stage 2: BERT Classification (CodeBERT)

Fine-tunes CodeBERT for detailed classification:

- Pre-trained on code understanding tasks
- Data augmentation (random cropping)
- Class-balanced training
- Optional adversarial training (FGM)

## 🔧 Advanced Features

### Multi-Model Ensemble

```python
# Edit config.yaml
pipeline:
  ensemble:
    enabled: true
    weights: [0.5, 0.5]  # LLM, BERT
```

### Training Optimizations

- **Mixed Precision Training** (FP16)
- **Gradient Accumulation**
- **Class Weighting** for imbalanced data
- **Cosine Learning Rate Scheduling**
- **Time Budget Control** for long experiments

### Data Augmentation

- Random cropping (head/tail/middle)
- Configurable crop strategies
- Two-view inference (averaging predictions)

## 📈 Evaluation Metrics

The pipeline computes:

- Accuracy
- Macro/Weighted F1 Score
- Per-class Precision/Recall
- Confusion Matrix
- Misclassification Analysis

Results are saved in:

- `results/metrics.json` - All metrics
- `results/confusion_matrix.png` - Visualization
- `results/per_class_metrics.png` - Per-class scores
- `results/misclassifications.csv` - Error analysis

## 🛠️ Customization

### Adding New Language Detector

```python
from src.language_detector import LanguageDetector

class CustomDetector(LanguageDetector):
    def detect_single(self, code_snippet):
        # Custom implementation
        pass
```

### Custom BERT Model

Edit `configs/config.yaml`:

```yaml
models:
  bert_classifier:
    model_name: "your-custom-bert-model"
    num_labels: 8
```

## 📝 Example Usage

```python
from src.pipeline import CodeClassificationPipeline
from src.utils import load_config
import pandas as pd

# Load configuration
config = load_config('configs/config.yaml')

# Initialize pipeline
pipeline = CodeClassificationPipeline(config)
pipeline.initialize_stages()

# Load test data
test_df = pd.read_parquet('data/raw/test.parquet')

# Run prediction
results = pipeline.predict(test_df, checkpoint_path='models/checkpoints/bert_best.pt')

# Save results
results.to_csv('results/predictions.csv', index=False)
```

## 🔬 Research Features

### Supported Languages

- **JavaScript** - Web development
- **PHP** - Server-side scripting
- **Java** - Enterprise applications
- **Python** - Data science, ML, general purpose
- **C#** - .NET framework
- **C++** - Systems programming, performance-critical
- **Go** - Cloud-native applications
- **C** - Low-level programming

### Model Architectures

1. **Qwen-2.5-Coder-1.5B-Instruct**
   - Lightweight LLM optimized for code
   - Fast inference
   - Strong few-shot capabilities

2. **CodeBERT**
   - Pre-trained on code corpus
   - Bidirectional context understanding
   - Fine-tuned for classification

## 📊 Expected Performance

Based on typical datasets:

- **Accuracy**: 85-95%
- **Macro F1**: 82-93%
- **Training Time**: 2-4 hours (on single GPU)
- **Inference Speed**: ~100 samples/second

## 🏆 Competition Results

### Leaderboard Rankings

| Rank | Team                | Score   | Entries | Last Submit |
| ---- | ------------------- | ------- | ------- | ----------- |
| 6   | NguyenDinhHaDuong   | 0.87066 | 22      | 43s         |
 
**Competition Type**: Code Language Classification  
**Evaluation Metric**: Macro F1 Score  
**Date**: February 2026

## 🐛 Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce batch size in config.yaml
training:
  batch_size: 8 # Try smaller values
  gradient_accumulation_steps: 2 # Increase to maintain effective batch size
```

### Long Training Time

```yaml
# Enable time budget control
training:
  time_budget_hours: 6
  safe_margin_min: 15
```

### Protobuf Warnings

These are non-fatal and automatically handled by the pipeline.

## 📚 References

- [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)

## 📄 License

This project is provided as-is for research and educational purposes.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Happy Coding!** 🎉
