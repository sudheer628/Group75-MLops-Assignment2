"""
Training script for Cats vs Dogs classifier.
Includes training loop, evaluation, and MLflow logging.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.dataset import (
    IDX_TO_CLASS,
    create_dataloaders,
)
from src.models.cnn import create_model


# Default hyperparameters
DEFAULT_CONFIG = {
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 10,
    "dropout": 0.5,
    "weight_decay": 1e-4,
    "num_workers": 0,
}


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a dataset.
    
    Returns:
        Tuple of (average_loss, accuracy, all_labels, all_predictions)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    avg_loss = running_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1_score": f1_score(y_true, y_pred, average="weighted"),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path) -> None:
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Cat", "Dog"],
        yticklabels=["Cat", "Dog"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    save_path: Path,
) -> None:
    """Create and save training curves plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, "b-", label="Train")
    ax1.plot(epochs, val_losses, "r-", label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, "b-", label="Train")
    ax2.plot(epochs, val_accs, "r-", label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train(
    config: Dict = None,
    data_dir: str = "data/processed",
    output_dir: str = "models",
    experiment_name: str = "cats_dogs_classification",
    use_mlflow: bool = True,
    max_samples: int = None,
) -> Dict:
    """
    Main training function.
    
    Args:
        config: Training configuration (uses DEFAULT_CONFIG if None)
        data_dir: Path to processed data
        output_dir: Path to save model and artifacts
        experiment_name: MLflow experiment name
        use_mlflow: Whether to log to MLflow
        max_samples: If set, limit dataset to this many samples per split (for quick testing)
        
    Returns:
        Dict with training results
    """
    config = config or DEFAULT_CONFIG.copy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        max_samples=max_samples,
    )
    
    # Create model
    print("Creating model...")
    model = create_model(num_classes=2, dropout=config["dropout"])
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    
    # Setup MLflow
    if use_mlflow:
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()
        mlflow.log_params(config)
    
    # Training loop
    print("Starting training...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(1, config["epochs"] + 1):
        print(f"\nEpoch {epoch}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Log to MLflow
        if use_mlflow:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, step=epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"  Saved best model (val_acc: {val_acc:.4f})")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt"))
    test_loss, test_acc, y_true, y_pred = evaluate(
        model, test_loader, criterion, device
    )
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics["test_loss"] = test_loss
    print(f"\nTest Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Save artifacts
    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        output_dir / "training_curves.png"
    )
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save class mapping
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump(IDX_TO_CLASS, f, indent=2)
    
    # Log to MLflow
    if use_mlflow:
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output_dir / "best_model.pt"))
        mlflow.log_artifact(str(output_dir / "confusion_matrix.png"))
        mlflow.log_artifact(str(output_dir / "training_curves.png"))
        mlflow.log_artifact(str(output_dir / "metrics.json"))
        mlflow.log_artifact(str(output_dir / "class_mapping.json"))
        mlflow.end_run()
    
    print("\nTraining complete!")
    return {"metrics": metrics, "best_val_acc": best_val_acc}


if __name__ == "__main__":
    train()
