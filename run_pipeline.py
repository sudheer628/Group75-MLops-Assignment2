"""
Main script to run the full ML pipeline.
Usage: python run_pipeline.py [--skip-download] [--epochs N]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(description="Run Cats vs Dogs ML Pipeline")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download (use existing raw data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for classifier head (default: 3e-4)",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=3e-5,
        help="Learning rate for backbone during fine-tuning (default: 3e-5)",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="mobilenet_v3_small",
        choices=["simple_cnn", "mobilenet_v3_small", "efficientnet_b0"],
        help="Model architecture (default: mobilenet_v3_small)",
    )
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=4,
        help="Number of epochs to train head with frozen backbone (default: 4)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers (default: 2)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use subset of data for fast testing",
    )
    args = parser.parse_args()

    # Step 1: Prepare data
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)
    
    from src.data.prepare_data import prepare_dataset
    stats = prepare_dataset(skip_download=args.skip_download)

    # Step 2: Train model
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)
    
    from src.training.train import train, DEFAULT_CONFIG
    
    config = DEFAULT_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.lr
    config["backbone_learning_rate"] = args.backbone_lr
    config["architecture"] = args.architecture
    config["freeze_epochs"] = args.freeze_epochs
    config["num_workers"] = args.num_workers
    
    # Quick mode uses subset
    max_samples = 500 if args.quick else None
    
    results = train(
        config=config,
        use_mlflow=not args.no_mlflow,
        max_samples=max_samples,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"Test metrics: {results['metrics']}")


if __name__ == "__main__":
    main()
