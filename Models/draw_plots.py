import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def draw_metrics_plot(train_dic, val_dic, model_version, path):
    metrics_names = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Kappa', 'MCC', 'ROC AUC']

    train_metrics = [train_dic['loss'], train_dic['accuracy'], train_dic['precision'], train_dic['recall'], train_dic['f1'],
                    train_dic['kappa'], train_dic['mcc'], train_dic['roc_auc']]

    val_metrics = [val_dic['loss'], val_dic['accuracy'], val_dic['precision'], val_dic['recall'], val_dic['f1'],
                    val_dic['kappa'], val_dic['mcc'], val_dic['roc_auc']]

    for train_metric, val_metric, metric_name in zip(train_metrics, val_metrics, metrics_names):
        plt.figure(figsize = (8, 5))
        plt.plot(train_metric, label = f'{model_version}. Train {metric_name}', color = 'blue')
        plt.plot(val_metric, label = f'{model_version}. Validation {metric_name}', color = 'red')
        # plt.plot(test_metric, label = f'{model_version}. Test {metric_name}', color = 'red', linestyle = '-', alpha = 1)
        plt.title(f'{model_version} {metric_name} over epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{path}{metric_name}_metrics_plot.png')
        # plt.show()


def draw_roc_curve(y, predictions, model_version, path):
    test_fpr, test_tpr, _ = roc_curve(y[:, 1], predictions[:, 1])

    plt.figure(figsize = (8, 5))
    plt.plot(test_fpr, test_tpr, color = 'red', label = 'Test ROC Curve')
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), c = 'black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_version}. Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}roc_curve_plot.png')
    # plt.show()


def draw_precision_recall_curve(y, predictions, model_version, path):
    train_precision, train_recall, _ = precision_recall_curve(y[:, 1], predictions[:, 1])

    plt.figure(figsize = (8, 5))
    plt.plot(train_recall, train_precision, color = 'blue', label = 'Train Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_version}Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{path}precision_recall_curve_plot.png')
    # plt.show()


def draw_confusion_matrix(y, predictions, model_version, path):
    conf_matrix = confusion_matrix(np.argmax(y, axis = 1), np.argmax(predictions, axis = 1))

    plt.figure(figsize = (8, 5))
    sns.heatmap(conf_matrix, annot = True, cmap = 'Blues', fmt = 'g')
    plt.title(f'{model_version} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'{path}confusion_matrix_plot.png')
    # plt.show()

