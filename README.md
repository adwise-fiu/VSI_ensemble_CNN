# Ensemble CNN Video Source Identification

This repository contains the source code for ensemble learning CNN video Source Identification. VISION dataset is required to run the experiment without any changes in code. 

1. Download videos from [VISION](https://lesc.dinfo.unifi.it/VISION/) keeping its ooriginal structure. 

## Prepare Train and Test dataset
Dataset folder contains the code to extract the frames and prepare train and test dataset.

1. To extract the I-frames from each video and save it locally run dataset/frames_extraction/extract_video_frames.py

```
python3 dataset/frames_extraction/extract_video_frames.py --vision_dataset_path="absolute/path/to/VISION/dataset" --frames_output_path="absolute/path/to/save/extracted/i-frames"
```

2. Run dataset/create_train_test_dataset.py to split extracted frames into test and training datasets

```
python3 dataset/create_train_test_dataset.py --frames_input_dataset="absolute/path/to/directory/with/extracted/frames" --test_train_dataset_folder="absolute/path/to/save/train/test/dataset"
```

3. Run dataset/patch_extractor.py to split extracted frames into test and training datasets. It is required to run it twice to extract patches for train dataset and for test dataset.

```
python3 dataset/patch_extractor.py --test_train_dataset_folder="absolute/path/to/directory/with/train/test/dataset" --sectors_output_folder="absolute/path/to/save/extracted/patches" --test_train="test"
```
```
python3 dataset/patch_extractor.py --test_train_dataset_folder="absolute/path/to/directory/with/train/test/dataset" --sectors_output_folder="absolute/path/to/save/extracted/patches" --test_train="train"
```