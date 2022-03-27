import os
import tensorflow as tf
from CNN_base_learners.cnn_network import ConstrainedLayer
from ensemble_data_generator import DataGeneratorEnsemble
from prepare_dataset_for_ensemble_net import DataSetGeneratorForEnsembleModel
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class EnsembleCNN:
    def __init__(self, quadrant_1_model_path, quadrant_2_model_path, quadrant_3_model_path, quadrant_4_model_path,
                    ds_path_quadrant_1, ds_path_quadrant_2, ds_path_quadrant_3, ds_path_quadrant_4,
                    path_to_folder):
        self.quadrant_1_model_path = quadrant_1_model_path
        self.quadrant_2_model_path = quadrant_2_model_path 
        self.quadrant_3_model_path = quadrant_3_model_path  
        self.quadrant_4_model_path = quadrant_4_model_path
        self.ds_path_quadrant_1 = ds_path_quadrant_1
        self.ds_path_quadrant_2 = ds_path_quadrant_2
        self.ds_path_quadrant_3 = ds_path_quadrant_3
        self.ds_path_quadrant_4 = ds_path_quadrant_4
        self.path_to_folder = path_to_folder
        self.__check_model_path()

    def __check_model_path(self):
        if not self.quadrant_1_model_path.endswith('.h5'):
            raise ValueError('!!! Incorrect path to quadrant 1 model file !!!')
        if not self.quadrant_2_model_path.endswith('.h5'):
            raise ValueError('!!! Incorrect path to quadrant 2 model file !!!')
        if not self.quadrant_3_model_path.endswith('.h5'):
            raise ValueError('!!! Incorrect path to quadrant 3 model file !!!')
        if not self.quadrant_4_model_path.endswith('.h5'):
            raise ValueError('!!! Incorrect path to quadrant 4 model file !!!')

    def get_files_and_labels(test_dictionary):
        label_key = "class_label"
        frame_path = "quadrant_1_patch"
        actual_encoded_list = [v for list in test_dictionary for k, v in list.items() if k == label_key]
        frames_list = [v for list in test_dictionary for k, v in list.items() if k == frame_path]
        return actual_encoded_list, frames_list

    def load_model(self, model_path):
        return tf.keras.models.load_model(model_path, custom_objects={
                                                     'ConstrainedLayer': ConstrainedLayer})

    def predict_model(self, model, ds_path, quadrant, num_classes):
        generator = DataGeneratorEnsemble(patches_path_dict=ds_path, quadrant=quadrant, num_classes=num_classes, to_fit=False, shuffle=False)
        predictions = model.predict(generator)
        return predictions

    def handle_difference_cause_by_batch_predictions(self, acttual_labels, predicted_labels):
        array_difference = len(acttual_labels) - len(predicted_labels)
        for i in range(array_difference):
            acttual_labels.pop()
        return np.array(acttual_labels)

    def calculate_quadrant_model_accuracy(self, predictions, actual_labels, quadrant):
        predicted_labels = np.argmax(predictions, axis=1)
        acc = accuracy_score(predicted_labels, actual_labels)
        print(f"The accuracy for quadrant {quadrant} is {acc}")

    def compute_average_voting(self, quadrant_1_predictions, quadrant_2_predictions, quadrant_3_predictions, quadrant_4_predictions, num_classes):
        combined_predictions_average = np.empty((len(quadrant_1_predictions), num_classes))
        for i in range(len(combined_predictions_average)):
            normalized_acc_1 = quadrant_1_predictions[i]
            normalized_acc_2 = quadrant_2_predictions[i]
            normalized_acc_3 = quadrant_3_predictions[i]
            normalized_acc_4 = quadrant_4_predictions[i]
            combined_predictions_average[i, ...] = (normalized_acc_1 + normalized_acc_2 + normalized_acc_3 + normalized_acc_4)/4

        return combined_predictions_average


    def run_ensemble(self):
        quadrant_1_model = self.load_model(self.quadrant_1_model_path)
        quadrant_2_model = self.load_model(self.quadrant_2_model_path)
        quadrant_3_model = self.load_model(self.quadrant_3_model_path)
        quadrant_4_model = self.load_model(self.quadrant_4_model_path)

        data_factory = DataSetGeneratorForEnsembleModel(input_dir_patchs=self.ds_path_quadrant_1)
        test_ds = data_factory.create_test_ds_4_quadrants_after_selection(self.ds_path_quadrant_1, self.ds_path_quadrant_2, self.ds_path_quadrant_3, self.ds_path_quadrant_4)
        actual_encoded_labels, frames_list = self.get_files_and_labels(test_ds)

        num_classes = len(data_factory.get_class_names())

        predictions_quadrant_1 = self.predict_model(quadrant_1_model, test_ds, 1, num_classes)
        predictions_quadrant_2 = self.predict_model(quadrant_2_model, test_ds, 2, num_classes)
        predictions_quadrant_3 = self.predict_model(quadrant_3_model, test_ds, 3, num_classes)
        predictions_quadrant_4 = self.predict_model(quadrant_4_model, test_ds, 4, num_classes)

        true_labels = self.handle_difference_cause_by_batch_predictions(actual_encoded_labels, predictions_quadrant_1)

        self.calculate_quadrant_model_accuracy(predictions_quadrant_1, true_labels, 1)
        self.calculate_quadrant_model_accuracy(predictions_quadrant_2, true_labels, 2)
        self.calculate_quadrant_model_accuracy(predictions_quadrant_3, true_labels, 3)
        self.calculate_quadrant_model_accuracy(predictions_quadrant_4, true_labels, 4)


        average_voting_predictions = np.argmax(self.compute_average_voting(predictions_quadrant_1, predictions_quadrant_2, predictions_quadrant_3, predictions_quadrant_4, num_classes), axis=1)

        acc_final = accuracy_score(average_voting_predictions, true_labels)
        print(f'Accuracy after average voting {acc_final}')

        self.save_precitions_labels(self.path_to_folder, true_labels, average_voting_predictions, frames_list)
        
    def save_precitions_labels(self, path, true_labels, predicted_labels, frames_list):
        full_path_to_file = os.path.join(path, "statistics")
        if not os.path.exists(full_path_to_file):
            os.makedirs(full_path_to_file)
        file_path = os.path.join(full_path_to_file, "ensemble_predictions.csv")
        actual_labels = np.argmax(true_labels, axis = 1)
        data_results = pd.DataFrame(list(zip(frames_list, actual_labels, predicted_labels)), columns=["File", "True Label", "Predicted Label"])
        data_results.to_csv(file_path, index=False) 
        