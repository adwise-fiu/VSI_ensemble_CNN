import os
import random
import argparse
from collections import namedtuple
from pathlib import Path
import time
import cv2
import numpy as np

parser = argparse.ArgumentParser(
    description='Extract patches for frames',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--test_train_dataset_folder', type=str, required=True, help='Path to directory with test and train datasets')
parser.add_argument('--quadrants_output_folder', type=str, required=True, help='Path to directory to save patches extracted per quadrant')
parser.add_argument('--test_train', type=str, required=True, help='Specify train or test dataset. Options: test, train')


def listdir_nohidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def get_patches(img_data, std_threshold, max_num_patches):
    patches = []

    # Default patches is returned when no patches are found with a Std.Dev. lower than the threshold
    default_patch_std = np.array([float('inf'), float('inf'), float('inf')])

    default_patch = None
    cropped_pathes = []

    patch = namedtuple('WindowSize', ['width', 'height'])(128, 128)
    stride = namedtuple('Strides', ['width_step', 'height_step'])(128, 128)
    image = namedtuple('ImageSize', ['width', 'height'])(img_data.shape[1], img_data.shape[0])
    default_patch_std = np.std(img_data.reshape(-1, 3), axis=0)
            
    for row_idx in range(patch.height, image.height, stride.height_step):
        for col_idx in range(patch.width, image.width, stride.width_step):
            cropped_img = img_data[(row_idx - patch.height):row_idx, (col_idx - patch.width):col_idx]
            patch_std = np.std(cropped_img.reshape(-1, 3), axis=0)
            if np.prod(np.less_equal(patch_std, std_threshold)):
                patches.append(cropped_img)
            elif np.prod(np.less_equal(patch_std, default_patch_std)):
                default_patch_std = patch_std
                default_patch = cropped_img
                cropped_pathes.append(cropped_img)
    if len(patches) < 10:
        for i in range(len(cropped_pathes)):
            if len(patches) > max_num_patches:
                break
            patches.append(cropped_pathes[i])
    if len(patches) > max_num_patches:
        random.seed(999)
        indices = random.sample(range(len(patches)), max_num_patches)
        patches = [patches[x] for x in indices]
    
    return patches

def crop_image_into_four_quadrants(img):
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    left1 = img[:, :width_cutoff]
    right1 = img[:, width_cutoff:]

    height_left = left1.shape[0]
    height_cutoff_left = height_left // 2
    first_quadrant = left1[:height_cutoff_left, :]
    second_quadrant = left1[height_cutoff_left:, :]
    
    # start vertical devide image
    height_right = img.shape[0]
    # Cut the image in half
    height_cutoff_right = height_right // 2
    third_quadrant = right1[:height_cutoff_right, :]
    forth_quadrant = right1[height_cutoff_right:, :]
    
    return first_quadrant, second_quadrant, third_quadrant, forth_quadrant


def save_patches(patches, source_img_path, destination_dir):
    for patch_id, patch in enumerate(patches, 1):
        img_name = source_img_path.stem + "_P-number" +'_{}'.format(str(patch_id).zfill(3)) + source_img_path.suffix
        img_path = destination_dir.joinpath(img_name)
        cv2.imwrite(str(img_path), patch * 255.0)


def main(source_data_dir, destination_patches_top_rigth, destination_patches_bottom_rigth, destination_patches_top_left, destination_patches_bottom_left):
    device_num_patches_dict_top_rigth = {}
    device_num_patches_dict_bottom_rigth = {}
    device_num_patches_dict_top_left = {}
    device_num_patches_dict_bottom_left = {}
    devices = source_data_dir.glob("*")
    if not destination_patches_top_rigth.exists():
        os.makedirs(str(destination_patches_top_rigth), exist_ok=True)
    if not destination_patches_bottom_rigth.exists():
        os.makedirs(str(destination_patches_bottom_rigth), exist_ok=True)
    if not destination_patches_top_left.exists():
        os.makedirs(str(destination_patches_top_left), exist_ok=True)
    if not destination_patches_bottom_left.exists():
        os.makedirs(str(destination_patches_bottom_left), exist_ok=True)

    t_start = time.time()
    for device in devices:
        image_paths = device.glob("*")
        destination_device_dir_pathches_top_rigth = destination_patches_top_rigth.joinpath(device.name)
        destination_device_dir_pathches_bottom_rigth = destination_patches_bottom_rigth.joinpath(device.name)
        destination_device_dir_pathches_top_left = destination_patches_top_left.joinpath(device.name)
        destination_device_dir_pathches_bottom_left = destination_patches_bottom_left.joinpath(device.name)

        # The following if-else construct makes sense on running multiple instances of this method
        if destination_device_dir_pathches_top_rigth.exists():
            continue
        else:
            os.makedirs(str(destination_device_dir_pathches_top_rigth), exist_ok=True)
        if destination_device_dir_pathches_bottom_rigth.exists():
            continue
        else:
            os.makedirs(str(destination_device_dir_pathches_bottom_rigth), exist_ok=True)
        if destination_device_dir_pathches_top_left.exists():
            continue
        else:
            os.makedirs(str(destination_device_dir_pathches_top_left), exist_ok=True)
        if destination_device_dir_pathches_bottom_left.exists():
            continue
        else:
            os.makedirs(str(destination_device_dir_pathches_bottom_left), exist_ok=True)    

        num_patches_top_right = 0
        num_patches_bottom_right = 0
        num_patches_top_left = 0
        num_patches_bottom_left = 0
        
        for image_path in image_paths:
            # For now, we only want to extract frames from original videos
            # if "WA" in image_path.stem or "YT" in image_path.stem:
            #     continue
            
            img = cv2.imread(str(image_path))
            img = np.float32(img) / 255.0

            quadrant_1, quadrant2, quadrant3, quadrant4 = crop_image_into_four_quadrants(img)
            
            pathes_1st_quadrant = get_patches(img_data=quadrant_1, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
            pathes_2nd_quadrant = get_patches(img_data=quadrant2, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
            pathes_3rd_quadrant = get_patches(img_data=quadrant3, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)
            pathes_4th_quadrant = get_patches(img_data=quadrant4, std_threshold=np.array([0.02, 0.02, 0.02]), max_num_patches=15)

            num_patches_top_right += len(pathes_1st_quadrant)
            num_patches_bottom_right += len(pathes_2nd_quadrant)
            num_patches_top_left += len(pathes_3rd_quadrant)
            num_patches_bottom_left += len(pathes_4th_quadrant)

            save_patches(pathes_1st_quadrant, image_path, destination_device_dir_pathches_top_rigth)
            save_patches(pathes_2nd_quadrant, image_path, destination_device_dir_pathches_bottom_rigth)
            save_patches(pathes_3rd_quadrant, image_path, destination_device_dir_pathches_top_left)
            save_patches(pathes_4th_quadrant, image_path, destination_device_dir_pathches_bottom_left)

        device_num_patches_dict_top_rigth[device.name] = num_patches_top_right
        device_num_patches_dict_bottom_rigth[device.name] = num_patches_bottom_right
        device_num_patches_dict_top_left[device.name] = num_patches_top_left
        device_num_patches_dict_bottom_left[device.name] = num_patches_bottom_left
        print(f"{device.name} | {num_patches_top_right} patches for 1st quadrant ({int(time.time() - t_start)} sec.)")
        print(f"{device.name} | {num_patches_bottom_right} patches for 2nd quadrant ({int(time.time() - t_start)} sec.)")
        print(f"{device.name} | {num_patches_top_left} patches for 3rd quadrant ({int(time.time() - t_start)} sec.)")
        print(f"{device.name} | {num_patches_bottom_left} patches for 4th quadrant ({int(time.time() - t_start)} sec.)")

    return device_num_patches_dict_top_rigth, device_num_patches_dict_bottom_rigth, device_num_patches_dict_top_left, device_num_patches_dict_bottom_left

if __name__ == "__main__":
    args = parser.parse_args()
    test_train_dataset_folder = args.test_train_dataset_folder
    quadrants_output_folder = args.quadrants_output_folder
    test_train = args.test_train
    input_frames_path = test_train_dataset_folder + "/" + test_train
    images_per_device = Path(input_frames_path)

    quadrant_1_path = quadrants_output_folder + "/quadrant_1/" + test_train
    quadrant_2_path = quadrants_output_folder + "/quadrant_2/" + test_train
    quadrant_3_path = quadrants_output_folder + "/quadrant_3/" + test_train
    quadrant_4_path = quadrants_output_folder + "/quadrant_4/" + test_train
    patches_per_device_top_rigth = Path(quadrant_1_path)
    patches_per_device_bottom_right = Path(quadrant_2_path)
    patches_per_device_top_left = Path(quadrant_3_path)
    patches_per_device_bottom_left = Path(quadrant_4_path)


    device_patch_dict_1st_quadrant, device_patch_dict_2nd_quadrant, device_patch_dict_3rd_quadrant, device_patch_dict_4th_quadrant = main(images_per_device, patches_per_device_top_rigth, patches_per_device_bottom_right, patches_per_device_top_left, patches_per_device_bottom_left)
