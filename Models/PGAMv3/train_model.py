import mlflow
import numpy as np
import mlflow.keras
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score

from Models.PGAMv3.get_training_data import get_training_data
from Models.find_files import find_files
from Models.read_fasta_file import read_fasta_file
from Models.parse_gtf_file import parse_gtf_file
from Models.encode_sequence import encode_sequence
from Models.draw_plots import *

def train_model(model, epochs, encoded_sequences_array, exon_array, n_window, n_samples_per_epoch, n_times, batch_size, nucleotide_codes):
    mlflow.set_tracking_uri("./Models/PGAMv3/mlflowRuns")
    mlflow.set_experiment("PGAMv3")

    mlflow.log_param("n_window", n_window)
    mlflow.log_param("n_samples_per_epoch", n_samples_per_epoch)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

    tensorboard_callback = TensorBoard(log_dir = './Models/PGAMv3/TensorBoardRuns', histogram_freq = n_times)

    train_history = {'loss' : [], 'accuracy' : [], 'precision' : [], 'recall' : [],
        'f1' : [], 'kappa' : [], 'mcc' : [], 'roc_auc' : []
    }

    val_history = {'loss' : [], 'accuracy' : [], 'precision' : [], 'recall' : [],
        'f1' : [], 'kappa' : [], 'mcc' : [], 'roc_auc' : []
    }

    global_train_history = {
        'loss' : [], 'accuracy' : [], 'precision' : [], 'recall' : [],
        'f1' : [], 'kappa' : [], 'mcc' : [], 'roc_auc' : []
    }

    global_val_history = {
        'loss' : [], 'accuracy' : [], 'precision' : [], 'recall' : [],
        'f1' : [], 'kappa' : [], 'mcc' : [], 'roc_auc' : []
    }

    for i in range(epochs):
        print(f"Global Epoch: {i + 1}")
        for j, (X_feature_1, X_feature_2, y_target) in enumerate(get_training_data(encoded_sequences_array, exon_array, n_window, n_samples_per_epoch, nucleotide_codes)):
            print(f"Local Epoch: {j + 1}")
            y_target_one_hot = to_categorical(y_target, num_classes = 2)
            X_1_train, X_1_val, X_2_train, X_2_val,  y_train, y_val = train_test_split(X_feature_1, X_feature_2, y_target_one_hot, train_size = 0.8, random_state = 1212)
            history = model.fit([X_1_train, X_2_train], y_train, epochs = n_times, batch_size = batch_size, validation_data = [[X_1_val, X_2_val], y_val], callbacks = [tensorboard_callback])
            
            train_history['loss'].append(np.mean(history.history['loss']))
            train_predictions = model.predict([X_1_train, X_2_train])
            train_predictions_classes = np.argmax(train_predictions, axis = 1)
            train_history['accuracy'].append(accuracy_score(np.argmax(y_train, axis = 1), train_predictions_classes))
            train_history['precision'].append(precision_score(np.argmax(y_train, axis = 1), train_predictions_classes))
            train_history['recall'].append(recall_score(np.argmax(y_train, axis = 1), train_predictions_classes))
            train_history['f1'].append(f1_score(np.argmax(y_train, axis = 1), train_predictions_classes))
            train_history['kappa'].append(cohen_kappa_score(np.argmax(y_train, axis = 1), train_predictions_classes))
            train_history['mcc'].append(matthews_corrcoef(np.argmax(y_train, axis = 1), train_predictions_classes))
            try:
                train_history['roc_auc'].append(roc_auc_score(y_train, train_predictions))
            except Exception as e:
                train_history['roc_auc'].append(0)
            
            val_history['loss'].append(np.mean(history.history['val_loss']))
            val_predictions = model.predict([X_1_val, X_2_val])
            val_predictions_classes = np.argmax(val_predictions, axis = 1)
            val_history['accuracy'].append(accuracy_score(np.argmax(y_val, axis = 1), val_predictions_classes))
            val_history['precision'].append(precision_score(np.argmax(y_val, axis = 1), val_predictions_classes))
            val_history['recall'].append(recall_score(np.argmax(y_val, axis = 1), val_predictions_classes))
            val_history['f1'].append(f1_score(np.argmax(y_val, axis = 1), val_predictions_classes))
            val_history['kappa'].append(cohen_kappa_score(np.argmax(y_val, axis = 1), val_predictions_classes))
            val_history['mcc'].append(matthews_corrcoef(np.argmax(y_val, axis = 1), val_predictions_classes))
            try:
                val_history['roc_auc'].append(roc_auc_score(y_val, val_predictions))
            except Exception as e:
                val_history['roc_auc'].append(0)


        for it in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'kappa', 'mcc', 'roc_auc']:
            global_train_history[it].append(np.sum(train_history[it]) / j)
            global_val_history[it].append(np.sum(val_history[it]) / j)
            
            train_history[it].clear()
            val_history[it].clear()


    for it in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'kappa', 'mcc', 'roc_auc']:
        mlflow.log_metric(f"train_{it}", global_train_history[it][-1])
        mlflow.log_metric(f"val_{it}", global_val_history[it][-1])


    fasta_test, gtf_test = find_files('./', 'test_sample', 'test_sample')

    predictions = []
    y_true = []

    for k, (X_feature_1, X_feature__2, y_target) in enumerate(get_training_data(fasta_test, gtf_test, n_window, n_samples_per_epoch, nucleotide_codes)):
        predicted = model.predict([X_feature_1, X_feature__2])

        predictions.append(predicted)
        y_true.append(to_categorical(y_target, num_classes = 2))


    predictions = np.concatenate(predictions)
    y_true = np.concatenate(y_true)

    draw_metrics_plot(global_train_history, global_val_history, "Model v3.", "./Moedls/PGAMv2/reports/")
    draw_roc_curve(y_true, predictions, "Model v3.", "./Moedls/PGAMv3/reports/")
    draw_precision_recall_curve(y_true, predictions, "Model v3.", "./Moedls/PGAMv3/reports/")
    draw_confusion_matrix(y_true, predictions, "Model v3.", "./Moedls/PGAMv3/reports/")

    mlflow.keras.log_model(model, "PGAMv3")
    mlflow.end_run()

    model.save("./Models/PGAMv3/reports/PGAMv3.h5")

    return model

