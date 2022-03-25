import numpy as np
import cv2
from tensorflow.keras.utils import Sequence



class DataGeneratorEnsemble(Sequence):
    def __init__(self,patches_path_dict, quadrant, num_classes, batch_size=32, to_fit=True, dim=(480,800,3), shuffle=True):
        self.patches_path_dict = patches_path_dict
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dim = dim
        self.quadrant = quadrant
        self.list_IDS = list(range(0, len(patches_path_dict)))
        self.shuffle = shuffle
        self.on_epoch_end()   

    def __len__(self):
        return int(np.floor(len(self.patches_path_dict)) / self.batch_size)     

    
    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDS[k] for k in indexes]
        quadrant_1_batch, quadrant_2_batch, quadrant_3_batch, quadrant_4_batch, labels_batch = self.__generate_frames_ds__(list_IDs_temp)

        if self.to_fit == True:
            return [quadrant_1_batch, quadrant_2_batch, quadrant_3_batch, quadrant_4_batch], labels_batch
        else:
            if self.quadrant == 1:
                return quadrant_1_batch
            elif self.quadrant == 2:
                return quadrant_2_batch
            elif self.quadrant == 3:
                return quadrant_3_batch
            elif self.quadrant == 4:
                return quadrant_4_batch


    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDS))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_frames_ds__(self, list_IDs_temp):
        quadrant_1_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        quadrant_2_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        quadrant_3_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        quadrant_4_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        labels_ds = np.empty((self.batch_size, self.num_classes), dtype=np.uint8)
        for i, id in enumerate(list_IDs_temp):
            key = "item_ID"
            val = id
            item = next((d for d in self.patches_path_dict if d.get(key) == val), None)
            
            quadrant_1 = self.__get_image__(item["quadrant_1_patch"])
            quadrant_2 = self.__get_image__(item["quadrant_2_patch"])
            quadrant_3 = self.__get_image__(item["quadrant_3_patch"])
            quadrant_4 = self.__get_image__(item["quadrant_4_patch"])
            label = item["class_label"]
            quadrant_1_ds[i, ...] = quadrant_1
            quadrant_2_ds[i, ...] = quadrant_2
            quadrant_3_ds[i, ...] = quadrant_3
            quadrant_4_ds[i, ...] = quadrant_4
            labels_ds[i, ...] = label
            
        return quadrant_1_ds, quadrant_2_ds, quadrant_3_ds, quadrant_4_ds, labels_ds


    #read image and resize it to (480,800, 3() and dt.float32 type)
    def __get_image__(self, image_path):
        img = cv2.imread(image_path)   
        return img