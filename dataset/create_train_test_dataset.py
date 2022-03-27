import os
import random
import argparse
import cv2

parser = argparse.ArgumentParser(
    description='Create Train and Test Datasets from frames dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--frames_input_dataset', type=str, required=True, help='Path to the directory containing extractd frames')
parser.add_argument('--test_train_dataset_folder', type=str, required=True, help='Path to directory to save test and train datasets')

def listdir_nohidden(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def get_video_compression_types(video_name):
    video_types = [video_name]
    
    for category in ['flat', 'indoor', 'outdoor']:
        if category in video_name:
            WA = video_name.replace(category, f"{category}WA")
            YT = video_name.replace(category, f"{category}YT")
            video_types.extend([WA, YT])
            return video_types

def copy_frames(src_path, dest_path, original_videos, device):
    train_frames = 0

    for video in original_videos:
        video_variations = get_video_compression_types(video)
        
        for video_variation in video_variations:
            if 'flat' in video_variation: 
                video_dest_path = os.path.join(src_path, '__flat__')
            if 'indoor' in video_variation: 
                video_dest_path = os.path.join(src_path, '__indoor__')
            if 'outdoor' in video_variation: 
                video_dest_path = os.path.join(src_path, '__outdoor__')
            vid_path = os.path.join(video_dest_path, video_variation)
            # print(vid_path)
            if os.path.exists(vid_path):
                video_name = os.path.basename(os.path.normpath(vid_path))
                frames = listdir_nohidden(vid_path)
                new_video_name = device + "_V_" + video_name.split("_V_")[1]

                sorted_frames = []
                # print(vid_path)
                if len(frames) > 15:
                    random.seed(999)
                    indices = random.sample(range(len(frames)), 15)
                    sorted_frames = [frames[x] for x in indices]
                else:
                    sorted_frames = frames
                print(f"for video {video}, selected frames: {len(sorted_frames)}")
                for frame in sorted_frames:
                    frame_src_path = os.path.join(vid_path, frame)
                    image = cv2.imread(frame_src_path)
                    file_name = "frame_number_"+str(train_frames)+ "_vid_name_" + new_video_name + ".jpg"
                    cv2.imwrite(os.path.join(dest_path, file_name), image)
                    train_frames += 1

                
if __name__ == "__main__":
    args = parser.parse_args()
    input_frames_dir = args.frames_input_dataset
    output_frames_dir = args.test_train_dataset_folder

    if not os.path.exists(output_frames_dir):
            os.makedirs(output_frames_dir)

    train_frames_dir = os.path.join(output_frames_dir, "train")
    test_frames_dir = os.path.join(output_frames_dir, "test")

    if not os.path.exists(train_frames_dir):
        os.mkdir(train_frames_dir)
    if not os.path.exists(test_frames_dir):
        os.mkdir(test_frames_dir)

    devices = [device for device in listdir_nohidden(input_frames_dir)]

    device_train_v = {}
    device_test_v = {}

    for device in devices:
        # if device in DATASET_DEVICES:
            d_src_path = os.path.join(input_frames_dir, device)
            d_dest_train_path = os.path.join(train_frames_dir, device)
            d_dest_test_path = os.path.join(test_frames_dir, device)

            if not os.path.exists(d_dest_train_path):
                os.mkdir(d_dest_train_path)
            if not os.path.exists(d_dest_test_path):
                os.mkdir(d_dest_test_path)

            flat_vids_dir = os.path.join(d_src_path, '__flat__')
            indoor_vids_dir = os.path.join(d_src_path, '__indoor__')
            outdoor_vids_dir = os.path.join(d_src_path, '__outdoor__')

            flat_vids = [v for v in listdir_nohidden(flat_vids_dir) if
                            os.path.isdir(os.path.join(flat_vids_dir, v)) and "_flat_" in v]
            indoor_vids = [v for v in listdir_nohidden(indoor_vids_dir) if
                            os.path.isdir(os.path.join(indoor_vids_dir, v)) and "_indoor_" in v]
            outdoor_vids = [v for v in listdir_nohidden(outdoor_vids_dir) if
                            os.path.isdir(os.path.join(outdoor_vids_dir, v)) and "_outdoor_" in v]

            num_original_vids = len(flat_vids) + len(indoor_vids) + len(outdoor_vids)
            
            num_flat_test_vids= int(len(flat_vids)*0.4)
            num_flat_train_vids = len(flat_vids) - num_flat_test_vids
            num_indoor_test_vids= int(len(indoor_vids)*0.4)
            num_indoor_train_vids = len(indoor_vids) - num_indoor_test_vids
            num_outdoor_test_vids= int(len(outdoor_vids)*0.4)
            num_outdoor_train_vids = len(outdoor_vids) - num_outdoor_test_vids

            num_test_vids = num_flat_test_vids + num_indoor_test_vids + num_outdoor_test_vids
            num_train_vids = num_flat_train_vids + num_indoor_train_vids + num_outdoor_train_vids

            print(f"\n{device} | Total videos: {num_original_vids}, train: {num_train_vids}, test: {num_test_vids}\n")

            random.shuffle(flat_vids)
            random.shuffle(indoor_vids)
            random.shuffle(outdoor_vids)

            train_vids = []
            test_vids = []

            train_vids.extend(flat_vids[0:1]) 
            test_vids.extend(flat_vids[1:2])

            del flat_vids[0:2]  # Remove the used flat vids
            
            train_vids.extend(indoor_vids[0:1])
            test_vids.extend(indoor_vids[1:2])

            del indoor_vids[0:2]

            train_vids.extend(outdoor_vids[0:1])
            test_vids.extend(outdoor_vids[1:2])

            del outdoor_vids[0:2]

            unused_flat = []
            unused_indoor = []
            unused_outdoor = []
            unused_flat.extend(flat_vids)
            unused_indoor.extend(indoor_vids)
            unused_outdoor.extend(outdoor_vids)

            random.shuffle(unused_flat)
            random.shuffle(unused_indoor)
            random.shuffle(unused_outdoor)

            num_remaining_train_flat = num_flat_train_vids - int(len(train_vids)/3)
            num_remaining_train_indoor = num_indoor_train_vids - int(len(train_vids)/3)
            num_remaining_train_outdoor = num_outdoor_train_vids - int(len(train_vids)/3)

            num_remaining_test_flat = num_flat_test_vids - int(len(test_vids)/3)
            num_remaining_test_indoor = num_indoor_test_vids - int(len(test_vids)/3)
            num_remaining_test_outdoor = num_outdoor_test_vids - int(len(test_vids)/3)

            for i in range(num_remaining_train_flat):
                train_vids.append(unused_flat[i])

            del unused_flat[0:num_remaining_train_flat]
            
            for i in range(num_remaining_train_indoor):
                train_vids.append(unused_indoor[i])

            del unused_indoor[0:num_remaining_train_indoor]

            for i in range(num_remaining_train_outdoor):
                train_vids.append(unused_outdoor[i])

            del unused_outdoor[0:num_remaining_train_outdoor]

            for i in range(num_remaining_test_flat):
                test_vids.append(unused_flat[i])
            
            del unused_flat[0:num_remaining_test_flat]

            for i in range(num_remaining_test_indoor):
                test_vids.append(unused_indoor[i])
            
            del unused_indoor[0:num_remaining_test_indoor]

            for i in range(num_remaining_test_outdoor):
                test_vids.append(unused_outdoor[i])
            
            del unused_outdoor[0:num_remaining_test_outdoor]

            combined = set(train_vids).intersection(test_vids)
            if len(combined) > 0:
                print("Error! The following video(s) occur in both test and unused set:")
                for item in combined:
                    print(f"{item}")

                raise ValueError("Error! Videos occur in both train and test set!")

            copy_frames(d_src_path, d_dest_train_path, train_vids, device)
            copy_frames(d_src_path, d_dest_test_path, test_vids, device)
            print(f"{device} | Total number of original videos: {num_original_vids}")
            print(f"{device} | Train ({len(train_vids)}): {train_vids}")
            print(f"{device} | Test ({len(test_vids)}): {test_vids}\n")