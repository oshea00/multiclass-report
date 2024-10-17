import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
import argparse

# Function to load the dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    if data.shape[1] != 2:
        raise ValueError("The CSV file must contain exactly two columns for predicted and expected values.")
    predicted = data.iloc[:, 0].astype(str)
    expected = data.iloc[:, 1].astype(str)
    return predicted, expected

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Multi-Class Confusion Matrix")
    plt.show()

# Function to print F1 scores
def print_f1_scores(y_true, y_pred, class_names):
    report = classification_report(y_true, y_pred, labels=class_names, target_names=class_names)
    print("\nClassification Report (F1 Scores):")
    print(report)

# Main function to run the script
def main():
    parser = argparse.ArgumentParser(description='Load prediction data, calculate confusion matrix, and plot it.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing predicted and expected pairs.')
    args = parser.parse_args()

    try:
        y_pred, y_true = load_data(args.csv_file)
        class_names = np.unique(np.concatenate((y_pred, y_true)))  # Get unique class names
        plot_confusion_matrix(y_true, y_pred, class_names)
        print_f1_scores(y_true, y_pred, class_names)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
