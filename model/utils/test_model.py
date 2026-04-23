import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchvision.transforms import v2

from sklearn.metrics import classification_report


def test_model(model_type, checkpoint_path, test_loader):
    # Load checkpoint data
    checkpoint = torch.load(checkpoint_path)
    model = model_type()
    model.load_state_dict(checkpoint['state_dict'])
    train_means = checkpoint['means']
    train_stds = checkpoint['stds']

    # Set up inference transformations
    inference_transforms = v2.Compose([
        v2.Resize(128, antialias=True),
        v2.Normalize(mean=train_means, std=train_stds)
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            inputs = inference_transforms(inputs)
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

    # Print the classification report
    print(classification_report(y_true, y_pred, target_names=class_names, output_dict=False))

    # Plotting
    sorted_f1 = dict(sorted(f1_scores.items(), key=lambda item: item[1]))

    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(sorted_f1.values()), y=list(sorted_f1.keys()), palette="viridis", hue=list(sorted_f1.keys()),
                legend=False)
    plt.axvline(x=float(np.mean(list(f1_scores.values()))), color='red', linestyle='--',
                label=f'Mean F1: {np.mean(list(f1_scores.values())):.2f}')
    plt.title(f"Per-Class F1-Score for {type(model).__name__} (Sorted)")
    plt.xlabel("F1-Score")
    plt.ylabel("Climate Class")
    plt.legend()
    plt.show()
