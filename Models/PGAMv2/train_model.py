import mlflow
import numpy as np
import mlflow.keras
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score

from Models.PGAMv2.get_training_data import get_training_data
from Models.find_files import find_files
from Models.read_fasta_file import read_fasta_file
from Models.parse_gtf_file import parse_gtf_file
from Models.encode_sequence import encode_sequence
from Models.draw_plots import *

def get_test_data(nucleotide_codes):
    fasta_files, gtf_files = find_files('./', 'test_sample', 'test_sample')
    sequences = []
    for fasta_file in fasta_files:
        sequences.append(read_fasta_file(fasta_file))

    sequences_array = np.array([list(seq) for seq in sequences], dtype = 'S1')
    
    exons_info = {}
    exon_array = np.zeros_like(sequences_array, dtype = int)

    for i, gtf_file in enumerate(gtf_files):
        exons_info.update(parse_gtf_file(gtf_file))

        for gene_id, exon_positions in exons_info.items():
            for start, end in exon_positions:
                exon_array[i, start - 1 : end] = 1

        exons_info.clear()


    encoded_sequences = []
    for sequence in sequences:
        encoded_sequences.append(encode_sequence(sequence, nucleotide_codes))

    sequences.clear()
    encoded_sequences_array = np.array(encoded_sequences)

    return encoded_sequences_array, exon_array


def train_model(model, epochs, encoded_sequences_array, exon_array, n_window, n_samples_per_epoch, n_times, batch_size, nucleotide_codes):
    X_test, y_test = get_test_data(nucleotide_codes)

    mlflow.set_tracking_uri("./Models/PGAMv2/mlflowRuns")
    mlflow.set_experiment("PGAMv2")

    mlflow.log_param("n_window", n_window)
    mlflow.log_param("n_samples_per_epoch", n_samples_per_epoch)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("epochs", epochs)

    tensorboard_callback = TensorBoard(log_dir = './Models/PGAMv1/TensorBoardRuns', histogram_freq = n_times)

    train_history = {'loss' : [], 'accuracy' : [], 'precision' : [], 'recall' : [],
        'f1' : [], 'kappa' : [], 'mcc' : [], 'roc_auc' : []
    }

    val_history = {'loss' : [], 'accuracy' : [], 'precision' : [], 'recall' : [],
        'f1' : [], 'kappa' : [], 'mcc' : [], 'roc_auc' : []
    }

    test_history = {
        'loss' : [], 'accuracy' : [], 'precision' : [], 'recall' : [],
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

    global_test_history = {
        'loss' : [], 'accuracy' : [], 'precision' : [], 'recall' : [],
        'f1' : [], 'kappa' : [], 'mcc' : [], 'roc_auc' : []
    }

    for i in range(epochs):
        print(f"Global Epoch: {i + 1}")
        for j, (X_feature, y_target) in enumerate(get_training_data(encoded_sequences_array, exon_array, n_window, n_samples_per_epoch, nucleotide_codes)):
            print(f"Local Epoch: {j + 1}")
            y_target_one_hot = to_categorical(y_target, num_classes = 2)
            X_train, X_val, y_train, y_val = train_test_split(X_feature, y_target_one_hot, train_size = 0.8, random_state = 1212)
            history = model.fit(X_train, y_train, epochs = n_times, batch_size = batch_size, validation_data = [X_val, y_val], callbacks = [tensorboard_callback])
            
            train_history['loss'].append(np.mean(history.history['loss']))
            train_predictions = model.predict(X_train)
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
            val_predictions = model.predict(X_val)
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

        for k, (X_feature, y_target) in enumerate(get_training_data(X_test, y_test, n_window, n_samples_per_epoch, nucleotide_codes)):
            y_target_one_hot = to_categorical(y_target, num_classes = 2)
            history = model.evaluate(X_feature, y_target_one_hot)
            
            test_history['loss'].append(history[0])
            test_predictions = model.predict(X_feature)
            test_predictions_classes = np.argmax(test_predictions, axis = 1)
            test_history['accuracy'].append(accuracy_score(np.argmax(y_target_one_hot, axis = 1), test_predictions_classes))
            test_history['precision'].append(precision_score(np.argmax(y_target_one_hot, axis = 1), test_predictions_classes))
            test_history['recall'].append(recall_score(np.argmax(y_target_one_hot, axis = 1), test_predictions_classes))
            test_history['f1'].append(f1_score(np.argmax(y_target_one_hot, axis = 1), test_predictions_classes))
            test_history['kappa'].append(cohen_kappa_score(np.argmax(y_target_one_hot, axis = 1), test_predictions_classes))
            test_history['mcc'].append(matthews_corrcoef(np.argmax(y_target_one_hot, axis = 1), test_predictions_classes))
            try:
                test_history['roc_auc'].append(roc_auc_score(y_target, val_predictions))
            except Exception as e:
                test_history['roc_auc'].append(0)

        for it in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'kappa', 'mcc', 'roc_auc']:
            global_train_history[it].append(np.sum(train_history[it]) / j)
            global_val_history[it].append(np.sum(val_history[it]) / j)
            global_test_history[it].append(np.sum(test_history[it]) / k)
            
            train_history[it].clear()
            val_history[it].clear()
            test_history[it].clear()


    for it in ['loss', 'accuracy', 'precision', 'recall', 'f1', 'kappa', 'mcc', 'roc_auc']:
        mlflow.log_metric(f"train_{it}", global_train_history[it][-1])
        mlflow.log_metric(f"val_{it}", global_val_history[it][-1])
        mlflow.log_metric(f"test_{it}", global_test_history[it][-1])


    predictions = []
    y_true = []

    for k, (X_feature, y_target) in enumerate(get_training_data(X_test, y_test, n_window, n_samples_per_epoch, nucleotide_codes)):
        predicted = model.predict(X_feature)

        predictions.append(predicted)
        y_true.append(to_categorical(y_target, num_classes = 2))


    predictions = np.concatenate(predictions)
    y_true = np.concatenate(y_true)

    draw_metrics_plot(global_train_history, global_val_history, global_test_history, "Model v2.", "./Moedls/PGAMv2/reports/")
    draw_roc_curve(y_true, predictions, "Model v2.", "./Moedls/PGAMv2/reports/")
    draw_precision_recall_curve(y_true, predictions, "Model v2.", "./Moedls/PGAMv2/reports/")
    draw_confusion_matrix(y_true, predictions, "Model v2.", "./Moedls/PGAMv2/reports/")

    mlflow.keras.log_model(model, "PGAMv2")
    mlflow.end_run()

    model.save("./Models/PGAMv2/reports/PGAMv2.h5")

    return model

