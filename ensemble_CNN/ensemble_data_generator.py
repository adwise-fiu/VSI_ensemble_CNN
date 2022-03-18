import numpy as np
import cv2
from tensorflow.keras.utils import Sequence



class DataGeneratorEnsemble(Sequence):
    def __init__(self,patches_path_dict, sector, num_classes, batch_size=32, to_fit=True, dim=(480,800,3), shuffle=True):
        self.patches_path_dict = patches_path_dict
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.dim = dim
        self.sector = sector
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
        sector_1_batch, sector_2_batch, sector_3_batch, sector_4_batch, labels_batch = self.__generate_frames_ds__(list_IDs_temp)

        if self.to_fit == True:
            return [sector_1_batch, sector_2_batch, sector_3_batch, sector_4_batch], labels_batch
        else:
            if self.sector == 1:
                return sector_1_batch
            elif self.sector == 2:
                return sector_2_batch
            elif self.sector == 3:
                return sector_3_batch
            elif self.sector == 4:
                return sector_4_batch


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDS))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __generate_frames_ds__(self, list_IDs_temp):
        sector_1_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        sector_2_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        sector_3_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        sector_4_ds = np.empty((self.batch_size, 128, 128, 3), dtype=np.uint8)
        labels_ds = np.empty((self.batch_size, self.num_classes), dtype=np.uint8)
        for i, id in enumerate(list_IDs_temp):
            key = "item_ID"
            val = id
            item = next((d for d in self.patches_path_dict if d.get(key) == val), None)
            
            sector_1 = self.__get_image__(item["sector_1_patch"])
            sector_2 = self.__get_image__(item["sector_2_patch"])
            sector_3 = self.__get_image__(item["sector_3_patch"])
            sector_4 = self.__get_image__(item["sector_4_patch"])
            label = item["class_label"]
            sector_1_ds[i, ...] = sector_1
            sector_2_ds[i, ...] = sector_2
            sector_3_ds[i, ...] = sector_3
            sector_4_ds[i, ...] = sector_4
            labels_ds[i, ...] = label
            
        return sector_1_ds, sector_2_ds, sector_3_ds, sector_4_ds, labels_ds


    #read image and resize it to (480,800, 3() and dt.float32 type)
    def __get_image__(self, image_path):
        img = cv2.imread(image_path)   
        return img
