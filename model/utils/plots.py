import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix


def plot_per_class_f1(model, fold, dirname, test_loader, val_gpu_transforms):
    model.eval()
    y_true = []
    y_pred = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            inputs = val_gpu_transforms(inputs)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Calculate F1 Scores
    # These are the standard 30 Köppen-Geiger classes
    class_names = [
        'Af', 'Am', 'Aw', 'BWh', 'BWk', 'BSh', 'BSk',
        'Csa', 'Csb', 'Csc', 'Cfa', 'Cfb', 'Cfc', 'Cwa', 'Cwb', 'Cwc',
        'Dsa', 'Dsb', 'Dsc', 'Dsd', 'Dfa', 'Dfb', 'Dfc', 'Dfd', 'Dwa', 'Dwb', 'Dwc', 'Dwd',
        'ET', 'EF'
    ]

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    f1_scores = {k: v['f1-score'] for k, v in report.items() if k in class_names}

    # Plotting
    sorted_f1 = dict(sorted(f1_scores.items(), key=lambda item: item[1]))

    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(sorted_f1.values()), y=list(sorted_f1.keys()), palette="viridis", hue=list(sorted_f1.keys()),
                legend=False)
    plt.axvline(x=float(np.mean(list(f1_scores.values()))), color='red', linestyle='--',
                label=f'Mean F1: {np.mean(list(f1_scores.values())):.2f}')
    plt.title("Per-Class Testing F1-Score (Sorted)")
    plt.xlabel("F1-Score")
    plt.ylabel("Climate Class")
    plt.legend()
    os.makedirs(f'reports/{dirname}/fold{fold}', exist_ok=True)
    plt.savefig(f"reports/{dirname}/fold{fold}/f1_score.png")
    plt.show()


def plot_acc_loss(hist, fold, dirname):
    plt.figure(figsize=(12, 5))

    # Plot the losses
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, hist[4] + 1), hist[0], 'r-', label='Training Loss')
    plt.plot(np.arange(1, hist[4] + 1), hist[1], 'b-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot the accuracies
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, hist[4] + 1), hist[2], 'r-', label='Training Accuracy')
    plt.plot(np.arange(1, hist[4] + 1), hist[3], 'b-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs(f'reports/{dirname}/fold{fold}', exist_ok=True)
    plt.savefig(f"reports/{dirname}/fold{fold}/loss_accuracy.png")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, epoch, fold, dirname):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - Fold {fold} - Epoch {epoch}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    os.makedirs(f'reports/{dirname}/fold{fold}/cm', exist_ok=True)
    plt.savefig(f'reports/{dirname}/fold{fold}/cm/epoch{epoch}.png')
    plt.close()
