import numpy as np
import pathlib, csv
import os, random
from glob import glob


class DataSetGeneratorForBranchCNN:
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
    
    def create_train_dataset(self):

        train_input_patchs_file_names = np.array(glob(str(self.train_dir_patchs) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()
        random.shuffle(train_input_patchs_file_names)
        
        for i, file_path in enumerate(train_input_patchs_file_names):
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": i, "patch_path": file_path, "class_label": class_label}                        
            labeled_dictionary.append(ds_row)

        return labeled_dictionary

    def create_validation_dataset(self):

        validation_input_patchs_file_names = np.array(glob(str(self.test_dir_patchs) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()

        random.shuffle(validation_input_patchs_file_names)

        for i, file_path in enumerate(validation_input_patchs_file_names):
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": i, "patch_path": file_path, "class_label": class_label}
            labeled_dictionary.append(ds_row)

        return labeled_dictionary

    def create_test_dataset(self):

        validation_input_patchs_file_names = np.array(glob(str(self.test_dir_patchs) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()

        random.shuffle(validation_input_patchs_file_names)

        for i, file_path in enumerate(validation_input_patchs_file_names):
            class_label = self.determine_label(file_path)
            ds_row = {"item_ID": i, "patch_path": file_path, "class_label": class_label}
            labeled_dictionary.append(ds_row)

        return labeled_dictionary

    # def create_train_ds_4_sectors(self, train_path_ds_1, train_path_ds_2, train_path_ds_3, train_path_ds_4):
    #     train_input_patches_file_names = np.array(glob(str(os.path.join(train_path_ds_1, "train")) + "/**/*.jpg", recursive = True))
    #     labeled_dictionary = list()
    #     random.shuffle(train_input_patches_file_names)
    #     counter = 0
    #     for i, file_path in enumerate(train_input_patches_file_names):
    #         class_label = self.determine_label(file_path)
    #         sector_1_patch = file_path
    #         sector_2_patch = self.get_sector_file_name(file_path, os.path.join(train_path_ds_2, "train"))
    #         sector_3_patch = self.get_sector_file_name(file_path, os.path.join(train_path_ds_3, "train"))
    #         sector_4_patch = self.get_sector_file_name(file_path, os.path.join(train_path_ds_4, "train"))
    #         if len(sector_2_patch) < 1 or len(sector_3_patch) < 1 or len(sector_4_patch) < 1:
    #             continue
    #         ds_row = {"item_ID": counter, "sector_1_patch":sector_1_patch, "sector_2_patch": sector_2_patch, 
    #             "sector_3_patch":sector_3_patch, "sector_4_patch": sector_4_patch, "class_label": class_label}
    #         labeled_dictionary.append(ds_row)
    #         counter += 1
    #     print(len(labeled_dictionary))
    #     return labeled_dictionary

    # def create_test_ds_4_sectors(self, test_path_ds_1, test_path_ds_2, test_path_ds_3, test_path_ds_4):
    #     test_input_patches_file_names = np.array(glob(str(os.path.join(test_path_ds_1, "test"))+ "/**/*.jpg", recursive = True))
    #     labeled_dictionary = list()
    #     random.shuffle(test_input_patches_file_names)
    #     counter = 0
    #     for i, file_path in enumerate(test_input_patches_file_names):
    #         class_label = self.determine_label(file_path)
    #         sector_1_patch = file_path
    #         sector_2_patch = self.get_sector_file_name(file_path, os.path.join(test_path_ds_2, "test"))
    #         sector_3_patch = self.get_sector_file_name(file_path, os.path.join(test_path_ds_3, "test"))
    #         sector_4_patch = self.get_sector_file_name(file_path, os.path.join(test_path_ds_4, "test"))
    #         if len(sector_2_patch) < 1 or len(sector_3_patch) < 1 or len(sector_4_patch) < 1:
    #             continue
    #         ds_row = {"item_ID": counter, "sector_1_patch":sector_1_patch, "sector_2_patch": sector_2_patch, 
    #             "sector_3_patch":sector_3_patch, "sector_4_patch": sector_4_patch, "class_label": class_label}
    #         labeled_dictionary.append(ds_row)
    #         counter += 1
    #     return labeled_dictionary
    
    def select_pathches(self, selected_file_path, all_pathes):
        selected_patches = []
        sorted_pathes = list()
        with open(selected_file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                selected_patches.append(row["0"])
            for path in all_pathes:
                if path in selected_patches:
                    sorted_pathes.append(path)
        
        return sorted_pathes
            

    def normalize_sectors_selection(self, sector1_patches, sector2_patches, sector3_patches, sector4_patches):
        # final_patches_sector_1 = list()
        final_patches_sector_2 = list()
        final_patches_sector_3 = list()
        final_patches_sector_4 = list()

        final_paths_dict = list()
        
        for path in sector1_patches:
            sector2_path = ""
            sector3_path = ""
            sector4_path = ""
            file_path, file_name = os.path.split(file_path)
            _, corrected_name = file_name.split("frame")
            frame_number, video_name = corrected_name.split("_vid_name_")
            classes = self.get_class_names()
            for device in classes:
                if device in file_path:
                    device_path_name = device
                    break
            for path2 in sector2_patches:
                if video_name in path2 and frame_number in path2 and device_path_name in path2 and path2 not in final_patches_sector_2:
                    sector2_path = path2
                    break
            for path3 in sector3_patches:
                if video_name in path3 and frame_number in path3 and device_path_name in path3 and path3 not in final_patches_sector_3:
                    sector3_path = path3
                    break
            for path4 in sector4_patches:
                if video_name in path4 and frame_number in path4 and device_path_name in path4 and path4 not in final_patches_sector_4:
                    sector4_path = path4
                    break
            
            if len(sector2_path) < 1 or len(sector3_path) < 1 or len(sector4_path) < 1:
                continue
            final_patches_sector_2.append(sector2_path)
            final_patches_sector_3.append(sector3_path)
            final_patches_sector_4.append(sector4_path)
            class_label = self.determine_label(path)
            addValue = {"sector_1_patch":path, "sector_2_patch": sector2_path, 
                "sector_3_patch":sector3_path, "sector_4_patch": sector4_path, "class_label": class_label}
            final_paths_dict.append(addValue)
        return final_paths_dict

    def create_train_ds_4_sectors(self, train_path_ds_1, train_path_ds_2, train_path_ds_3, train_path_ds_4):
        train_input_patches_file_names_sector1 = np.array(glob(str(os.path.join(train_path_ds_1, "train")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_sector2 = np.array(glob(str(os.path.join(train_path_ds_2, "train")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_sector3 = np.array(glob(str(os.path.join(train_path_ds_3, "train")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_sector4 = np.array(glob(str(os.path.join(train_path_ds_4, "train")) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()
        selected_pathches_by_sectors = self.normalize_sectors_selection(train_input_patches_file_names_sector1, train_input_patches_file_names_sector2, train_input_patches_file_names_sector3, train_input_patches_file_names_sector4)
        random.shuffle(selected_pathches_by_sectors)
        for i, item in enumerate(selected_pathches_by_sectors):
            
            ds_row = {"item_ID": i, "sector_1_patch": item["sector_1_patch"], "sector_2_patch": item["sector_2_patch"], 
                "sector_3_patch": item["sector_3_patch"], "sector_4_patch": item["sector_4_patch"], "class_label": item["class_label"]}
            labeled_dictionary.append(ds_row)
        return labeled_dictionary

    def create_test_ds_4_sectors(self, train_path_ds_1, train_path_ds_2, train_path_ds_3, train_path_ds_4):
        train_input_patches_file_names_sector1 = np.array(glob(str(os.path.join(train_path_ds_1, "test")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_sector2 = np.array(glob(str(os.path.join(train_path_ds_2, "test")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_sector3 = np.array(glob(str(os.path.join(train_path_ds_3, "test")) + "/**/*.jpg", recursive = True))
        train_input_patches_file_names_sector4 = np.array(glob(str(os.path.join(train_path_ds_4, "test")) + "/**/*.jpg", recursive = True))
        labeled_dictionary = list()
        selected_pathches_by_sectors = self.normalize_sectors_selection(train_input_patches_file_names_sector1, train_input_patches_file_names_sector2, train_input_patches_file_names_sector3, train_input_patches_file_names_sector4)
        random.shuffle(selected_pathches_by_sectors)
        for i, item in enumerate(selected_pathches_by_sectors):
            
            ds_row = {"item_ID": i, "sector_1_patch": item["sector_1_patch"], "sector_2_patch": item["sector_2_patch"], 
                "sector_3_patch": item["sector_3_patch"], "sector_4_patch": item["sector_4_patch"], "class_label": item["class_label"]}
            labeled_dictionary.append(ds_row)
        return labeled_dictionary

    def get_sector_file_name(self, file_path, directory):
        file_path, file_name = os.path.split(file_path)
        remove_part, corrected_name = file_name.split("frame")
        frame_number, patch_name = corrected_name.split("_vid_name_")
        # first_split = path.split("vid_name_")[1]
        # video_name = first_split.split("_P")[0]
        classes = self.get_class_names()
        for device in classes:
            if device in file_path:
                device_path_name = device
                break
        input_patches_per_directory = np.array(glob(str(os.path.join(directory,device_path_name)) + "/**/*.jpg", recursive = True))
        
        sector_patch_file = ""
        for patch_file_path in input_patches_per_directory:
            if patch_name in patch_file_path and frame_number in patch_file_path:
                sector_patch_file = patch_file_path
                break
        return sector_patch_file