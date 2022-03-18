import os
import tensorflow as tf
from branch_CNN.cnn_network import ConstrainedLayer
from ensemble_CNN.ensemble_data_generator import DataGeneratorEnsemble
from prepare_dataset_for_ensemble_net import DataSetGeneratorForEnsembleModel
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def get_files_and_labels(test_dictionary):
    label_key = "class_label"
    frame_path = "sector_1_patch"
    actual_encoded_list = [v for list in test_dictionary for k, v in list.items() if k == label_key]
    frames_list = [v for list in test_dictionary for k, v in list.items() if k == frame_path]
    return actual_encoded_list, frames_list

sector_1_model_path ="/data/home/mveksler/experiment_one/sector_1/fm-e00156.h5"
sector_2_model_path ="/data/home/mveksler/experiment_one/sector_2/fm-e00127.h5"
sector_3_model_path ="/data/home/mveksler/experiment_one/sector_3/fm-e00124.h5"
sector_4_model_path ="/data/home/mveksler/experiment_one/sector_4/fm-e00178.h5"

sector_1_ds_path = "/data/home/mveksler/experiment_1_ten_devices_frames_1st_sector"
sector_2_ds_path = "/data/home/mveksler/experiment_1_ten_devices_frames_2nd_sector"
sector_3_ds_path = "/data/home/mveksler/experiment_1_ten_devices_frames_3rd_sector"
sector_4_ds_path = "/data/home/mveksler/experiment_1_ten_devices_frames_4th_sector"

sector_1_selected_path = "/Users/marynavek/Projects/video_identification_cnn/binary_results_sector_1_.csv"
sector_2_selected_path = "/Users/marynavek/Projects/video_identification_cnn/binary_results_sector_2_.csv"
sector_3_selected_path = "/Users/marynavek/Projects/video_identification_cnn/binary_results_sector_3_.csv"
sector_4_selected_path = "/Users/marynavek/Projects/video_identification_cnn/binary_results_sector_4_.csv"

labels = []
sector_1_model = tf.keras.models.load_model(sector_1_model_path, custom_objects={
                                                     'ConstrainedLayer': ConstrainedLayer})
sector_2_model = tf.keras.models.load_model(sector_2_model_path, custom_objects={
                                                     'ConstrainedLayer': ConstrainedLayer})
sector_3_model = tf.keras.models.load_model(sector_3_model_path, custom_objects={
                                                     'ConstrainedLayer': ConstrainedLayer})
sector_4_model = tf.keras.models.load_model(sector_4_model_path, custom_objects={
                                                     'ConstrainedLayer': ConstrainedLayer})

data_factory = DataSetGeneratorForEnsembleModel(input_dir_patchs=sector_1_ds_path)
test_ds = data_factory.create_test_ds_4_sectors_after_selection(sector_1_ds_path, sector_2_ds_path, sector_3_ds_path, sector_4_ds_path)
print(len(test_ds))
actual_encoded_labels_test, frames_list_test = get_files_and_labels(test_ds)

num_classes = len(data_factory.get_class_names())

print("predict 1st sector")
generator1 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=1, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_1 = sector_1_model.predict(generator1)

print("predict 2nd sector")
generator2 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=2, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_2 = sector_2_model.predict(generator2)

print("predict 3rdd sector")
generator3 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=3, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_3 = sector_3_model.predict(generator3)

print("predict 4th sector")
generator4 = DataGeneratorEnsemble(patches_path_dict=test_ds, sector=4, num_classes=num_classes, to_fit=False, shuffle=False)
predictsSector_4 = sector_4_model.predict(generator4)

array_difference = len(actual_encoded_labels_test) - len(predictsSector_1)
for i in range(array_difference):
    actual_encoded_labels_test.pop()
true_labels = np.array(actual_encoded_labels_test)


print("predictions")
# print(predictions)
# print(predictions.shape)
actual_labels1 = np.argmax(true_labels, axis = 1)
print(np.shape(predictsSector_1), np.shape(true_labels))
predicted1 = np.argmax(predictsSector_1, axis = 1)
acc_arr_1 = np.equal(predicted1, actual_labels1)
correct_1 = 0
for i in acc_arr_1:
    if i:
        correct_1 += 1
acc1 = correct_1/len(predicted1)
# acc1 = accuracy_score(predicted1, actual_labels1)
print(f"accuracy score sector_1 {acc1}")
predicted2 = np.argmax(predictsSector_2, axis = 1)
acc_arr_2 = np.equal(predicted2, actual_labels1)
correct_2 = 0
for i in acc_arr_2:
    if i:
        correct_2 += 1
acc2 = correct_2/len(predicted2)
# acc2 = accuracy_score(predicted2, actual_labels1)
print(f"accuracy score sector_2 {acc2}")
predicted3 = np.argmax(predictsSector_3, axis = 1)
acc_arr_3 = np.equal(predicted3, actual_labels1)
correct_3 = 0
for i in acc_arr_3:
    if i:
        correct_3 += 1
acc3 = correct_3/len(predicted3)
# acc3 = accuracy_score(predicted3, actual_labels1)
print(f"accuracy score sector_3 {acc3}")
predicted4 = np.argmax(predictsSector_4, axis = 1)
acc_arr_4 = np.equal(predicted4, actual_labels1)
correct_4 = 0
for i in acc_arr_4:
    if i:
        correct_4 += 1
acc4 = correct_4/len(predicted4)
# acc4 = accuracy_score(predicted4, actual_labels1)
print(f"accuracy score sector_4 {acc4}")

print("evaluate on test_ds\n")

combined_predictions_average = np.empty((len(predictsSector_1), num_classes))


for i in range(len(combined_predictions_average)):
    normalized_acc_1 = predictsSector_1[i]
    normalized_acc_2 = predictsSector_2[i]
    normalized_acc_3 = predictsSector_3[i]
    normalized_acc_4 = predictsSector_4[i]
    combined_predictions_average[i, ...] = (normalized_acc_1 + normalized_acc_2 + normalized_acc_3 + normalized_acc_4)/4

predicted_combined = np.argmax(combined_predictions_average, axis = 1)

acc_arr_comb = np.equal(predicted_combined, actual_labels1)
correct_comb = 0
for i in acc_arr_comb:
    if i:
        correct_comb += 1
acc_combined = correct_comb/len(predicted_combined)


# acc_combined = accuracy_score(predicted_combined, actual_labels1)
print(f"accuracy score combined {acc_combined}")

print("evaluate on train_ds\n")


output_base = "/data/home/mveksler/ensemble_results/experiment_one"
if not os.path.exists(output_base):
    os.makedirs(output_base)
output_file_path_total = os.path.join(output_base, "new_predictions_combined.csv")


# predicted_labels = np.argmax(predictions, axis = 1)
actual_labels = np.argmax(true_labels, axis = 1)
data_results = pd.DataFrame(list(zip(frames_list_test, actual_labels, predicted_combined)), columns=["File", "True Label", "Predicted Label"])
data_results.to_csv(output_file_path_total, index=False) 

# predicted_labels = np.argmax(predictions, axis = 1)
# # actual_labels = np.argmax(true_labels, axis = 1)
# data_results = pd.DataFrame(list(zip(frames_list_test, actual_labels, predicted_labels)), columns=["File", "True Label", "Predicted Label"])
# data_results.to_csv(output_file_path_rf, index=False) 

# pred_prob = rf.predict_proba(stackY)

# # roc curve for classes
# fpr = {}
# tpr = {}
# thresh ={}

# n_classes = num_classes

# fpr = dict()
# tpr = dict()
# roc_auc = dict()


# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], combined_predictions_average[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), combined_predictions_average.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # for i in range(n_classes):
# #   fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], predictions[:, i])
# #   plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
# #   print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))

# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# # Finally average it and compute AUC
# mean_tpr /= n_classes

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# # Plot all ROC curves
# plt.figure()
# plt.plot(
#     fpr["micro"],
#     tpr["micro"],
#     label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
#     color="deeppink",
#     linestyle=":",
#     linewidth=4,
# )

# plt.plot(
#     fpr["macro"],
#     tpr["macro"],
#     label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
#     color="navy",
#     linestyle=":",
#     linewidth=4,
# )

# colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", 'red', 'blueviolet', 'grey', 'black', "teal", "bisque"])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(
#         fpr[i],
#         tpr[i],
#         color=color,
#         lw=2,
#         label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
#     )

# plt.plot([0, 1], [0, 1], "k--", lw=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Some extension of Receiver operating characteristic to multiclass")
# plt.legend(loc="lower right")
# plt.show()