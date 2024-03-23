import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


def draw_metrics_plot(train_dic, val_dic, test_dic):
    metrics_names = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Kappa', 'MCC', 'ROC AUC']

    train_metrics = [train_dic['loss'], train_dic['accuracy'], train_dic['precision'], train_dic['recall'], train_dic['f1'],
                    train_dic['kappa'], train_dic['mcc'], train_dic['roc_auc']]

    val_metrics = [val_dic['loss'], val_dic['accuracy'], val_dic['precision'], val_dic['recall'], val_dic['f1'],
                    val_dic['kappa'], val_dic['mcc'], val_dic['roc_auc']]
    
    test_metrics = [test_dic['loss'], test_dic['accuracy'], test_dic['precision'], test_dic['recall'], test_dic['f1'],
                    test_dic['kappa'], test_dic['mcc'], test_dic['roc_auc']]

    for train_metric, val_metric, test_metric, metric_name in zip(train_metrics, val_metrics, test_metrics, metrics_names):
        plt.figure(figsize = (8, 5))
        plt.plot(train_metric, label = f'Train {metric_name}', color = 'green', linestyle = '--', alpha = 0.6)
        plt.plot(val_metric, label = f'Validation {metric_name}', color = 'blue', linestyle = ':', alpha = 0.8)
        plt.plot(test_metric, label = f'Test {metric_name}', color = 'red', linestyle = '-', alpha = 1)
        plt.title(f'{metric_name} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./Models/PGAMv1/reports/{metric_name}_metrics_plot.png')
        plt.show()


def draw_roc_curve(y, predictions):
    test_fpr, test_tpr, _ = roc_curve(y[:, 1], predictions[:, 1])

    plt.figure(figsize = (8, 5))
    plt.plot(test_fpr, test_tpr, color = 'red', label = 'Test ROC Curve')
    plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), c = 'black')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('./Models/PGAMv1/reports/roc_curve_plot.png')
    plt.show()


def draw_precision_recall_curve(y, predictions):
    train_precision, train_recall, _ = precision_recall_curve(y[:, 1], predictions[:, 1])

    plt.figure(figsize = (8, 5))
    plt.plot(train_recall, train_precision, color = 'blue', label = 'Train Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('./Models/PGAMv1/reports/precision_recall_curve_plot.png')
    plt.show()


def draw_confusion_matrix(y, predictions):
    conf_matrix = confusion_matrix(np.argmax(y, axis = 1), np.argmax(predictions, axis = 1))

    plt.figure(figsize = (8, 5))
    sns.heatmap(conf_matrix, annot = True, cmap = 'Blues', fmt = 'g')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('./Models/PGAMv1/reports/confusion_matrix_plot.png')
    plt.show()