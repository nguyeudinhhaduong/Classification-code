# Project Structure Documentation

## File Organization

```
code_classification_pipeline/
│
├── configs/                          # Configuration files
│   └── config.yaml                   # Main configuration
│
├── data/                             # Data directory
│   ├── raw/                          # Raw parquet/csv files
│   │   ├── train.parquet
│   │   ├── val.parquet
│   │   └── test.parquet
│   └── processed/                    # Preprocessed data
│       └── train_with_languages.parquet
│
├── models/                           # Model storage
│   ├── pretrained/                   # Pre-trained models cache
│   └── checkpoints/                  # Training checkpoints
│       └── bert_best.pt
│
├── src/                              # Source code modules
│   ├── __init__.py                   # Package initialization
│   ├── language_detector.py         # Stage 1: LLM language detection
│   ├── bert_classifier.py           # Stage 2: BERT classification
│   ├── pipeline.py                   # Pipeline orchestrator
│   ├── data_loader.py                # Data loading utilities
│   ├── utils.py                      # Helper functions
│   └── evaluator.py                  # Evaluation & metrics
│
├── notebooks/                        # Jupyter notebooks
│   └── exploration.ipynb             # Data exploration
│
├── results/                          # Output results
│   ├── metrics.json                  # Evaluation metrics
│   ├── confusion_matrix.png          # Confusion matrix plot
│   ├── per_class_metrics.png         # Per-class scores
│   ├── misclassifications.csv        # Error analysis
│   └── submission.csv                # Competition submission
│
├── logs/                             # Log files
│   ├── pipeline.log                  # Training logs
│   └── tensorboard/                  # TensorBoard logs
│
├── main.py                           # Main training script
├── demo.py                           # Quick demo script
├── requirements.txt                  # Python dependencies
├── README.md                         # Main documentation
├── .gitignore                        # Git ignore rules
└── PROJECT_STRUCTURE.md              # This file
```

## Module Descriptions

### Core Pipeline Modules

- **language_detector.py**: Uses Qwen LLM to detect programming language
  - Class: `LanguageDetector`
  - Key methods: `detect_single()`, `detect_batch()`

- **bert_classifier.py**: CodeBERT-based classification
  - Class: `BERTClassifier`
  - Key methods: `train()`, `predict()`, `evaluate()`

- **pipeline.py**: Orchestrates the complete workflow
  - Class: `CodeClassificationPipeline`
  - Key methods: `train()`, `predict()`, `evaluate()`

### Utility Modules

- **data_loader.py**: Data loading and preprocessing
  - Class: `DataLoader`
  - Handles parquet files, splits, preprocessing

- **utils.py**: Common utilities
  - Functions: `seed_everything()`, `load_config()`, `setup_logging()`

- **evaluator.py**: Metrics and visualization
  - Class: `Evaluator`
  - Generates reports, plots, analysis

## Workflow

### Training Pipeline

1. **Data Loading** (`data_loader.py`)
   - Load train/val/test data
   - Preprocess and clean
2. **Stage 1: Language Detection** (`language_detector.py`)
   - Optional preprocessing step
   - Adds language metadata
3. **Stage 2: BERT Training** (`bert_classifier.py`)
   - Fine-tune CodeBERT
   - Save checkpoints
4. **Evaluation** (`evaluator.py`)
   - Compute metrics
   - Generate visualizations

### Inference Pipeline

1. Load test data
2. Run language detection (optional)
3. Load BERT checkpoint
4. Generate predictions
5. Save submission file

## Configuration

All settings are centralized in `configs/config.yaml`:

- **Data paths**: Input/output locations
- **Model settings**: Architecture, hyperparameters
- **Training config**: Batch size, learning rate, epochs
- **Pipeline stages**: Enable/disable components

## Scripts

### Main Scripts

- `main.py`: Full training/evaluation pipeline

  ```bash
  python main.py --mode train
  python main.py --mode eval --checkpoint path/to/checkpoint
  python main.py --mode predict --checkpoint path/to/checkpoint
  ```

- `demo.py`: Quick demonstration
  ```bash
  python demo.py
  ```

### Notebooks

- `exploration.ipynb`: Data analysis and visualization

## Data Flow

```
Raw Data (parquet)
    ↓
Data Loader (preprocessing)
    ↓
Stage 1: Language Detection (Qwen LLM)
    ↓
Stage 2: BERT Classification (CodeBERT)
    ↓
Predictions & Evaluation
    ↓
Results (CSV, plots, metrics)
```

## Dependencies

See `requirements.txt` for full list. Key libraries:

- PyTorch (deep learning)
- Transformers (pre-trained models)
- Pandas (data handling)
- Scikit-learn (metrics)
- Matplotlib/Seaborn (visualization)

## Best Practices

1. **Data**: Keep raw data in `data/raw/`, processed in `data/processed/`
2. **Models**: Save checkpoints in `models/checkpoints/`
3. **Results**: All outputs go to `results/`
4. **Logs**: Check `logs/` for debugging
5. **Config**: Edit `config.yaml` instead of hardcoding values

## Extending the Pipeline

### Add New Stage

1. Create module in `src/`
2. Add stage config in `config.yaml`
3. Update `pipeline.py` to include stage

### Custom Model

1. Implement model class (inherit from base)
2. Update config with model path
3. Modify classifier to use new model

### New Metrics

1. Add metric functions to `evaluator.py`
2. Update `config.yaml` metrics list
3. Metrics auto-computed in evaluation
