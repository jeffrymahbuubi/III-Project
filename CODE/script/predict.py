import argparse
from model import PretrainedAlexNet2D, PretrainedAlexNet2DPrunedL1, PretrainedAlexNet2DPrunedL2, PretrainedMobileNetV2, PretrainedMobileNetV2WithL1Pruning, PretrainedMobileNetV2WithL2Pruning
from dataset import get_data, get_data_kfold, get_model_path
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report
import torch
from torch.utils.data import DataLoader


# Define a mapping dictionary for models
model_mapping = {
    0: lambda: PretrainedAlexNet2D(num_classes=8),
    1: lambda: PretrainedAlexNet2DPrunedL1(num_classes=8, pruning_amount=0.2),
    2: lambda: PretrainedAlexNet2DPrunedL2(num_classes=8, pruning_amount=0.2),
    3: lambda: PretrainedMobileNetV2(num_classes=8),
    4: lambda: PretrainedMobileNetV2WithL1Pruning(num_classes=8, pruning_amount=0.2),
    5: lambda: PretrainedMobileNetV2WithL2Pruning(num_classes=8, pruning_amount=0.2),
}

model_name = ["PretrainAlexNet", "PretrainAlexNetPruneL1", "PretrainAlexNetPruneL2",
              "PretrainMobileNetV2", "PretrainMobileNetV2PruneL1", "PretrainMobileNetV2PruneL2"]

def predict(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, num_classes: int = 8):
    model.eval()

    all_predictions = []
    all_labels = []
    total_inference_time = 0  # To track total inference time

    with torch.no_grad():
        for data, target in dataloader:
            # Start the timer for inference
            start_time = time.time()

            # Move data and labels to the device
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Stop the timer after inference
            end_time = time.time()
            total_inference_time += (end_time - start_time)

            # Assuming the model's output is logits; apply softmax for class probabilities
            probabilities = torch.softmax(output, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()

            # Collect predictions and true labels
            all_predictions.extend(predicted_labels)
            all_labels.extend(target.cpu().numpy())

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=list(range(num_classes)))

    # Calculate overall metrics
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    # Calculate class-wise metrics manually
    precision = {}
    recall = {}
    f1_score = {}
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp

        precision_cls = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_cls = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_cls = (2 * precision_cls * recall_cls) / (precision_cls + recall_cls) if (precision_cls + recall_cls) > 0 else 0

        precision[f"class_{i}"] = precision_cls
        recall[f"class_{i}"] = recall_cls
        f1_score[f"class_{i}"] = f1_cls

        total_precision += precision_cls
        total_recall += recall_cls
        total_f1 += f1_cls

    # Compute mean metrics across classes
    mean_precision = total_precision / num_classes
    mean_recall = total_recall / num_classes
    mean_f1 = total_f1 / num_classes

    # Calculate inference speed
    num_batches = len(dataloader)
    avg_inference_time_per_batch = total_inference_time / num_batches if num_batches > 0 else 0
    total_samples = len(all_labels)
    avg_inference_time_per_sample = total_inference_time / total_samples if total_samples > 0 else 0

    return {
        "predictions": all_predictions,
        "accuracy": accuracy,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "conf_matrix": conf_matrix,
        "class_precision": precision,
        "class_recall": recall,
        "class_f1": f1_score,
        "labels": all_labels,
        "total_inference_time": total_inference_time,
        "avg_inference_time_per_batch": avg_inference_time_per_batch,
        "avg_inference_time_per_sample": avg_inference_time_per_sample
    }


def main(args):
    # Ensure indices are valid
    if args.model_index not in model_mapping:
        raise ValueError(f"Invalid model_index {args.model_index}. Must be between 0 and {len(model_mapping) - 1}.")

    # Initialize the model dynamically
    model = model_mapping[args.model_index]()

    # Load the model weights
    state_dict = torch.load(get_model_path(args.model_index, model_name, args.model_dir))
    model.load_state_dict(state_dict)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Load data using k-fold cross-validation
    if args.k_fold:
        train_loader, val_loader = get_data_kfold(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            num_folds=args.num_folds,
            fold_index=args.fold_index,
            random_state=args.random_state,
        )
    else:
        train_loader, val_loader = get_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            test_size=args.test_size,
            num_workers=args.num_workers,
            random_state=args.random_state,
        )

    # Perform predictions
    results = predict(model, val_loader, device, num_classes=8)

    # Display overall results
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Mean Precision: {results['mean_precision']:.2%}")
    print(f"Mean Recall: {results['mean_recall']:.2%}")
    print(f"Mean F1: {results['mean_f1']:.2%}")

    # Display confusion matrix
    print("\nConfusion Matrix:")
    print(results['conf_matrix'])

    # Display metrics for each class
    print("\nClass-wise Metrics:")
    for cls in sorted(results['class_precision'].keys()):
        precision = results['class_precision'][cls] * 100
        recall = results['class_recall'][cls] * 100
        f1 = results['class_f1'][cls] * 100
        print(f"Class {cls}: Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1-Score: {f1:.2f}%")

    # Display inference time
    print("\nInference Time:")
    print(f"Total Inference Time: {results['total_inference_time']:.4f} seconds")
    print(f"Average Inference Time per Batch: {results['avg_inference_time_per_batch']:.4f} seconds")
    print(f"Average Inference Time per Sample: {results['avg_inference_time_per_sample']:.6f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions on a model using validation data.")

    # Model selection
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing model weights.")
    parser.add_argument("--model_index", type=int, required=True, help="Index of the model to use (0-5).")

    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")

    # k-Fold parameters
    parser.add_argument("--k_fold", action="store_true", help="Enable k-Fold Cross Validation.")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for k-Fold Cross Validation.")
    parser.add_argument("--fold_index", type=int, default=0, help="Index of the fold to use as validation set (0-indexed).")

    # Train-test split parameters
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of data to use as validation set.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()
    main(args)
