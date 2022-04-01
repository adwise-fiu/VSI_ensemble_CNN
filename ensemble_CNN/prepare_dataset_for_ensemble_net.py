import numpy as np
import pathlib, csv
import os, random
from glob import glob


class DataSetGeneratorForEnsembleModel:
    def __init__(self, input_dir_patchs=None, test_dir_suffix="", classes=None): 
        self.data_dir_patchs = pathlib.Path(input_dir_patchs)

        self.train_dir_patchs = pathlib.Path(os.path.join(self.data_dir_patchs, "train"))
        self.test_dir_patchs = pathlib.Path(os.path.join(self.data_dir_patchs, f"test{test_dir_suffix}"))
        
        self.device_types = np.array([item.name for item in self.train_dir_patchs.glob('*') if not item.name.startswith('.')])
        self.train_image_count = len(list(self.train_dir_patchs.glob('*/*.jpg')))
        self.test_image_count = len(list(self.test_dir_patchs.glob('*/*.jpg')))

        self.class_names = self.get_classes(classes)
        print(self.class_names)

    def get_classes(self, classes):
        if classes is not None:
            return classes
        else:
            class_names = sorted(self.train_dir_patchs.glob("*"))
            return np.array([x.name for x in class_names if not x.name.startswith('.')])            

    def get_class_names(self):
        return self.class_names

    def device_count(self):
        return len(self.class_names)

    def get_image_count(self, type="train"):
        if type == "train":
            return self.train_image_count
        else:
            return self.test_image_count

    def listdir_nonhidden(path):
        return [f for f in os.listdir(path) if not f.startswith('.')]

    def determine_label(self, file_path):
        classes = self.get_class_names()
        label_vector_lenght = self.device_count()
        label = np.zeros((label_vector_lenght,), dtype=int)
        classes.sort()
        for i, class_name in enumerate(classes):
            if class_name in file_path:
                label[i] = 1
        return label

    def normalize_quadrants_selection(self, quadrant1_patches, quadrant2_patches, quadrant3_patches, quadrant4_patches):
        final_patches_quadrant_2 = list()
        final_patches_quadrant_3 = list()
        final_patches_quadrant_4 = list()

        final_paths_dict = list()
        
        for path in quadrant1_patches:
            quadrant2_path = ""
            quadrant3_path = ""
            quadrant4_path = ""
            file_path, file_name = os.path.split(path)
            _, corrected_name = file_name.split("frame")
            frame_number, video_name = corrected_name.split("_vid_name_")
            classes = self.get_class_names()
            for device in classes:
                if device in file_path:
                    device_path_name = device
                    break
            for path2 in quadrant2_patches:
                if video_name in path2 and frame_number in path2 and device_path_name in path2 and path2 not in final_patches_quadrant_2:
                    quadrant2_path = path2
                    break
            for path3 in quadrant3_patches:
                if video_name in path3 and frame_number in path3 and device_path_name in path3 and path3 not in final_patches_quadrant_3:
                    quadrant3_path = path3
                    break
            for path4 in quadrant4_patches:
                if video_name in path4 and frame_number in path4 and device_path_name in path4 and path4 not in final_patches_quadrant_4:
                    quadrant4_path = path4
                    break
            
            if len(quadrant2_path) < 1 or len(quadrant3_path) < 1 or len(quadrant4_path) < 1:
                continue
            final_patches_quadrant_2.append(quadrant2_path)
            final_patches_quadrant_3.append(quadrant3_path)
            final_patches_quadrant_4.append(quadrant4_path)
            class_label = self.determine_label(path)
            addValue = {"quadrant_1_patch":path, "quadrant_2_patch": quadrant2_path, 
                "quadrant_3_patch":quadrant3_path, "quadrant_4_patch": quadrant4_path, "class_label": class_label}
            final_paths_dict.append(addValue)
        return final_paths_dict

    def create_train_ds_4_quadrants(self, train_path_ds_1, train_path_ds_2, train_path_ds_3, train_path_ds_4):
        train_input_patches_file_names_quadrant1 = np.array(glob(str(os.path.join(train_path_ds_1, "train")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_quadrant2 = np.array(glob(str(os.path.join(train_path_ds_2, "train")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_quadrant3 = np.array(glob(str(os.path.join(train_path_ds_3, "train")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_quadrant4 = np.array(glob(str(os.path.join(train_path_ds_4, "train")) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()
        selected_pathches_by_quadrants = self.normalize_quadrants_selection(train_input_patches_file_names_quadrant1, train_input_patches_file_names_quadrant2, train_input_patches_file_names_quadrant3, train_input_patches_file_names_quadrant4)
        random.shuffle(selected_pathches_by_quadrants)
        for i, item in enumerate(selected_pathches_by_quadrants):
            
            ds_row = {"item_ID": i, "quadrant_1_patch": item["quadrant_1_patch"], "quadrant_2_patch": item["quadrant_2_patch"], 
                "quadrant_3_patch": item["quadrant_3_patch"], "quadrant_4_patch": item["quadrant_4_patch"], "class_label": item["class_label"]}
            labeled_dictionary.append(ds_row)
        return labeled_dictionary

    def create_test_ds_4_quadrants(self, train_path_ds_1, train_path_ds_2, train_path_ds_3, train_path_ds_4):
        train_input_patches_file_names_quadrant1 = np.array(glob(str(os.path.join(train_path_ds_1, "test")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_quadrant2 = np.array(glob(str(os.path.join(train_path_ds_2, "test")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_quadrant3 = np.array(glob(str(os.path.join(train_path_ds_3, "test")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_quadrant4 = np.array(glob(str(os.path.join(train_path_ds_4, "test")) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()
        selected_pathches_by_quadrants = self.normalize_quadrants_selection(train_input_patches_file_names_quadrant1, train_input_patches_file_names_quadrant2, train_input_patches_file_names_quadrant3, train_input_patches_file_names_quadrant4)
        random.shuffle(selected_pathches_by_quadrants)
        for i, item in enumerate(selected_pathches_by_quadrants):
            
            ds_row = {"item_ID": i, "quadrant_1_patch": item["quadrant_1_patch"], "quadrant_2_patch": item["quadrant_2_patch"], 
                "quadrant_3_patch": item["quadrant_3_patch"], "quadrant_4_patch": item["quadrant_4_patch"], "class_label": item["class_label"]}
            labeled_dictionary.append(ds_row)
        return labeled_dictionary

    def get_quadrant_file_name(self, file_path, directory):
        file_path, file_name = os.path.split(file_path)
        remove_part, corrected_name = file_name.split("frame")
        frame_number, patch_name = corrected_name.split("_vid_name_")
        classes = self.get_class_names()
        for device in classes:
            if device in file_path:
                device_path_name = device
                break
        input_patches_per_directory = np.array(glob(str(os.path.join(directory,device_path_name)) + "/**/*.jpg", recursive = True))
        
        quadrant_patch_file = ""
        for patch_file_path in input_patches_per_directory:
            if patch_name in patch_file_path and frame_number in patch_file_path:
                quadrant_patch_file = patch_file_path
                break
        return quadrant_patch_file