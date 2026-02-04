"""
Main Training Script
Complete pipeline for training code classification models.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils import setup_logging, load_config, seed_everything, print_device_info
from src.data_loader import DataLoader
from src.pipeline import CodeClassificationPipeline
from src.evaluator import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Code Classification Pipeline')
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'eval', 'predict'],
        default='train',
        help='Running mode: train, eval, or predict'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (for eval/predict)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models/checkpoints',
        help='Output directory for models and results'
    )
    
    return parser.parse_args()


def train_pipeline(config, output_dir):
    """
    Train the complete pipeline.
    
    Args:
        config: Configuration dictionary
        output_dir: Output directory for models
    """
    # Setup
    seed_everything(config['project']['seed'])
    print_device_info()
    
    # Load data
    data_loader = DataLoader(config)
    train_df, val_df, test_df = data_loader.load_all_data()
    
    # Preprocess
    train_df = data_loader.preprocess_data(train_df)
    val_df = data_loader.preprocess_data(val_df)
    
    # Initialize pipeline
    pipeline = CodeClassificationPipeline(config)
    
    # Train
    pipeline.train(train_df, val_df, output_dir)
    
    print("\n✓ Training completed!")
    print(f"Models saved to: {output_dir}")


def evaluate_pipeline(config, checkpoint_path, output_dir):
    """
    Evaluate the pipeline on test data.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for results
    """
    # Load data
    data_loader = DataLoader(config)
    _, _, test_df = data_loader.load_all_data()
    test_df = data_loader.preprocess_data(test_df)
    
    # Get label mapping
    label_mapping = data_loader.get_label_mapping(test_df)
    
    # Initialize pipeline and evaluator
    pipeline = CodeClassificationPipeline(config)
    evaluator = Evaluator(label_mapping)
    
    # Evaluate
    metrics = pipeline.evaluate(test_df, checkpoint_path)
    
    # Generate report
    results = pipeline.predict(test_df, checkpoint_path)
    evaluator.generate_evaluation_report(
        test_df['label'].values,
        results['predicted_label'].values,
        df=test_df,
        output_dir=output_dir,
        prefix="test_"
    )
    
    print("\n✓ Evaluation completed!")
    print(f"Results saved to: {output_dir}")


def predict_pipeline(config, checkpoint_path, output_dir):
    """
    Run inference on test data.
    
    Args:
        config: Configuration dictionary
        checkpoint_path: Path to model checkpoint
        output_dir: Output directory for predictions
    """
    # Load data
    data_loader = DataLoader(config)
    test_df = data_loader.load_test_data()
    
    # Initialize pipeline
    pipeline = CodeClassificationPipeline(config)
    
    # Predict
    results = pipeline.predict(test_df, checkpoint_path)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    submission_file = output_path / config['output']['submission_file']
    
    # Prepare submission
    id_col = 'id' if 'id' in results.columns else 'ID'
    if id_col not in results.columns:
        results[id_col] = range(len(results))
    
    submission = results[[id_col, 'predicted_label']].copy()
    submission.columns = ['id', 'label']
    submission.to_csv(submission_file, index=False)
    
    print("\n✓ Prediction completed!")
    print(f"Submission saved to: {submission_file}")
    print(f"\nFirst 5 predictions:")
    print(submission.head())


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_config = config['logging']
    setup_logging(
        log_dir=log_config['log_dir'],
        log_file=log_config['log_file'],
        level=log_config['level']
    )
    
    print("=" * 60)
    print(f"Code Classification Pipeline - {args.mode.upper()} Mode")
    print("=" * 60)
    
    # Run appropriate mode
    if args.mode == 'train':
        train_pipeline(config, args.output_dir)
    
    elif args.mode == 'eval':
        if args.checkpoint is None:
            print("Error: --checkpoint required for eval mode")
            sys.exit(1)
        evaluate_pipeline(config, args.checkpoint, config['output']['results_dir'])
    
    elif args.mode == 'predict':
        if args.checkpoint is None:
            print("Error: --checkpoint required for predict mode")
            sys.exit(1)
        predict_pipeline(config, args.checkpoint, config['output']['results_dir'])
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
