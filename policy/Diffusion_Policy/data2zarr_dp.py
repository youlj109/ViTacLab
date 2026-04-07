import argparse
import json
import os
import shutil

import numpy as np
import zarr
from PIL import Image
from tqdm import tqdm
import copy
import pickle

def main():
    parser = argparse.ArgumentParser(
        description="Convert data to zarr format for diffusion policy"
    )
    parser.add_argument(
        "task_name",
        type=str,
        default="Fold_Dress",
        help="The name of the task (e.g., Fold_Dress)",
    )
    parser.add_argument(
        "train_data_num",
        type=int,
        default=200,
        help="Number of data to process (e.g., 200)",
    )
    args = parser.parse_args()
    
    task_name = args.task_name
    train_data_num = args.train_data_num
    
    current_abs_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(os.path.dirname(current_abs_dir))
    print("Project Root Dir : ", parent_dir)
    
    load_dir_parent = parent_dir + f"/data/rsl_rl/{task_name}"
    load_dir = sorted(os.listdir(load_dir_parent))[-1]
    load_dir = os.path.join(load_dir_parent, load_dir)
    print("Meta Data Load Dir : ", load_dir)
    
    save_dir = f"data/{task_name}_{train_data_num}.zarr"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    print("Save Dir : ", save_dir)
    
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")
    
    # ZARR datasets will be created dynamically during the first batch write
    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    
    # Batch processing settings
    batch_size = 200
    head_camera_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_count = 0
    current_batch = 0
    
    for current_ep in tqdm(range(train_data_num), desc=f"Processing {train_data_num} MetaData"):
        data = np.load(load_dir + f'/episode_{current_ep}.npz', allow_pickle=True)
        # for i in range(data_length-1):
            # head_camera_arrays.append(data[i]['camera_rgb_third_person_camera'])
            # state_arrays.append(data[i]['actions'])
            # action_arrays.append(data[i+1]['actions'])
        head_camera_arrays.append(data['camera_rgb'][:-1])
        state_arrays.append(data['joint_pos'][:-1])
        action_arrays.append(data['joint_pos'][1:])
        total_count += len(data['joint_pos']) - 1
        episode_ends_arrays.append(copy.deepcopy(total_count))
        
        # Write to ZARR if batch is full or if this is the last episode
        if (current_ep + 1) % batch_size == 0 or (current_ep + 1) == train_data_num:
            # Convert arrays to NumPy and format head_camera
            print("head_camera_arrays shape: ", np.concatenate(head_camera_arrays).shape)
            print("action_arrays shape: ", np.concatenate(action_arrays).shape)
            print("state_arrays shape: ", np.concatenate(state_arrays).shape)
            print("episode_ends_arrays shape: ", np.array(episode_ends_arrays).shape)
            head_camera_arrays = np.moveaxis(np.concatenate(head_camera_arrays), -1, 1)  # NHWC -> NCHW
            action_arrays = np.concatenate(action_arrays)
            state_arrays = np.concatenate(state_arrays)
            episode_ends_arrays = np.array(episode_ends_arrays)
            
            # Create datasets dynamically during the first write
            if current_batch == 0:
                zarr_data.create_dataset(
                    "head_camera",
                    shape=(0, *head_camera_arrays.shape[1:]),
                    chunks=(batch_size, *head_camera_arrays.shape[1:]),
                    dtype=head_camera_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "state",
                    shape=(0, state_arrays.shape[1]),
                    chunks=(batch_size, state_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "action",
                    shape=(0, action_arrays.shape[1]),
                    chunks=(batch_size, action_arrays.shape[1]),
                    dtype="float32",
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_meta.create_dataset(
                    "episode_ends",
                    shape=(0,),
                    chunks=(batch_size,),
                    dtype="int64",
                    compressor=compressor,
                    overwrite=True,
                )
            
            # Append data to ZARR datasets
            zarr_data["head_camera"].append(head_camera_arrays)
            zarr_data["state"].append(state_arrays)
            zarr_data["action"].append(action_arrays)
            zarr_meta["episode_ends"].append(episode_ends_arrays)
            
            print(
                f"Batch {current_batch + 1} written with {len(head_camera_arrays)} samples."
            )
            
            print(f"head_camera shape: {head_camera_arrays.shape}")
            print(f"state shape: {state_arrays.shape}")
            print(f"action shape: {action_arrays.shape}")
            print(f"episode_ends shape: {episode_ends_arrays.shape}")
            
            # Clear arrays for next batch
            head_camera_arrays = []
            action_arrays = []
            state_arrays = []
            episode_ends_arrays = []
            current_batch += 1
            

if __name__ == "__main__":
    main()