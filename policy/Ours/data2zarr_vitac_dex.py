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
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

def main():
    def pose_quat_to_pose6d(pose_array: np.ndarray, rot_tf: RotationTransformer) -> np.ndarray:
        pos = pose_array[:, :3]
        quat = pose_array[:, 3:7]
        rot6d = rot_tf.forward(quat).astype(pose_array.dtype, copy=False)
        return np.concatenate([pos, rot6d], axis=-1)

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
    parent_dir = os.path.dirname(current_abs_dir)
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
    camera_arrays = []
    camera_pos_arrays = []
    tac_rgb_arrays = []
    tac_pos_arrays = []
    tac_force_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    total_count = 0
    current_batch = 0
    quat_to_rot6d_tf = RotationTransformer(from_rep="quaternion", to_rep="rotation_6d")
    
    for current_ep in tqdm(range(train_data_num), desc=f"Processing {train_data_num} MetaData"):
        data = np.load(load_dir + f'/episode_{current_ep:04d}.npz', allow_pickle=True)
        data_length = len(data)
        # for i in range(data_length-1):
            # head_camera_arrays.append(data[i]['camera_rgb_third_person_camera'])
            # state_arrays.append(data[i]['actions'])
            # action_arrays.append(data[i+1]['actions'])
        camera_keys = sorted([k for k in data.files if k.endswith("camera")])
        if len(camera_keys) == 0:
            raise KeyError("No camera keys found in episode data.")

        camera_pos_keys = [f"{k}_pos" for k in camera_keys]
        missing_camera_pos_keys = [k for k in camera_pos_keys if k not in data.files]
        if missing_camera_pos_keys:
            raise KeyError(
                f"Missing camera position keys for cameras {camera_keys}: {missing_camera_pos_keys}"
            )

        camera_rgb = np.stack([data[k] for k in camera_keys], axis=1)
        camera_pos = np.stack([data[k].squeeze(1) for k in camera_pos_keys], axis=1)
        camera_arrays.append(camera_rgb)
        camera_pos_arrays.append(camera_pos)
        # left_hand_tac_arrays.append((data['tactile_rgb_image'].reshape(-1, 2, 320, 240, 3)[:-1, 0, :, :, :]).astype(np.uint8))
        # right_hand_tac_arrays.append((data['tactile_rgb_image'].reshape(-1, 2, 320, 240, 3)[:-1, 1, :, :, :]).astype(np.uint8))
        tactile_normal_force = data['tactile_normal_force']
        tactile_shear_force = data['tactile_shear_force']
        tactile_force = np.concatenate([tactile_normal_force, tactile_shear_force], axis=-1)
        
        tactile_rgb_image = data['tactile_rgb_image']
        tactile_pos = data['tactile_pos']
        
        tac_rgb_arrays.append(tactile_rgb_image)
        tac_pos_arrays.append(tactile_pos)
        tac_force_arrays.append(tactile_force)
        state_arrays.append(data['joint_pos'])
        action_arrays.append(data['action'])
        total_count += len(data['joint_pos'])
        episode_ends_arrays.append(copy.deepcopy(total_count))
        
        # Write to ZARR if batch is full or if this is the last episode
        if (current_ep + 1) % batch_size == 0 or (current_ep + 1) == train_data_num:
            # Convert arrays to NumPy and format camera
            print("camera_arrays shape: ", np.concatenate(camera_arrays).shape, "max: ", np.concatenate(camera_arrays).max(), "min: ", np.concatenate(camera_arrays).min())
            print("tac_rgb_arrays shape: ", np.concatenate(tac_rgb_arrays).shape, "max: ", np.concatenate(tac_rgb_arrays).max(), "min: ", np.concatenate(tac_rgb_arrays).min())
            print("tac_force_arrays shape: ", np.concatenate(tac_force_arrays).shape, "max: ", np.concatenate(tac_force_arrays).max(), "min: ", np.concatenate(tac_force_arrays).min())
            print("action_arrays shape: ", np.concatenate(action_arrays).shape, "max: ", np.concatenate(action_arrays).max(), "min: ", np.concatenate(action_arrays).min())
            print("state_arrays shape: ", np.concatenate(state_arrays).shape, "max: ", np.concatenate(state_arrays).max(), "min: ", np.concatenate(state_arrays).min())
            print("episode_ends_arrays shape: ", np.array(episode_ends_arrays).shape, "max: ", np.array(episode_ends_arrays).max(), "min: ", np.array(episode_ends_arrays).min())
            print("state_arrays shape: ", np.concatenate(state_arrays).shape)
            print("episode_ends_arrays shape: ", np.array(episode_ends_arrays).shape)
            print("camera_pos_arrays shape: ", np.concatenate(camera_pos_arrays).shape, "max: ", np.concatenate(camera_pos_arrays).max(), "min: ", np.concatenate(camera_pos_arrays).min())
            print("tac_pos_arrays shape: ", np.concatenate(tac_pos_arrays).shape, "max: ", np.concatenate(tac_pos_arrays).max(), "min: ", np.concatenate(tac_pos_arrays).min())
            camera_arrays = np.moveaxis(np.concatenate(camera_arrays), -1, 2)  # N2HWC -> N2CHW
            tac_rgb_arrays = np.moveaxis(np.concatenate(tac_rgb_arrays), -1, 2)  # N2HWC -> N2CHW
            tac_force_arrays = np.concatenate(tac_force_arrays)  # 这里不moveaxis，因为后续normalizer要在最后一维上做normalization
            camera_pos_arrays = np.concatenate(camera_pos_arrays)
            tac_pos_arrays = np.concatenate(tac_pos_arrays)
            camera_pos_shape = camera_pos_arrays.shape
            camera_pos_arrays = pose_quat_to_pose6d(
                camera_pos_arrays.reshape(-1, camera_pos_shape[-1]),
                quat_to_rot6d_tf
            ).reshape(camera_pos_shape[0], camera_pos_shape[1], -1)
            tac_pos_shape = tac_pos_arrays.shape
            tac_pos_arrays = pose_quat_to_pose6d(
                tac_pos_arrays.reshape(-1, tac_pos_shape[-1]),
                quat_to_rot6d_tf
            ).reshape(tac_pos_shape[0], tac_pos_shape[1], -1)
            action_arrays = np.concatenate(action_arrays)
            state_arrays = np.concatenate(state_arrays)
            episode_ends_arrays = np.array(episode_ends_arrays)
            
            # Create datasets dynamically during the first write
            if current_batch == 0:
                zarr_data.create_dataset(
                    "camera",
                    shape=(0, *camera_arrays.shape[1:]),
                    chunks=(batch_size, *camera_arrays.shape[1:]),
                    dtype=camera_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "camera_pos",
                    shape=(0, *camera_pos_arrays.shape[1:]),
                    chunks=(batch_size, *camera_pos_arrays.shape[1:]),
                    dtype=camera_pos_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "tac_pos",
                    shape=(0, *tac_pos_arrays.shape[1:]),
                    chunks=(batch_size, *tac_pos_arrays.shape[1:]),
                    dtype=tac_pos_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "tac_rgb",
                    shape=(0, *tac_rgb_arrays.shape[1:]),
                    chunks=(batch_size, *tac_rgb_arrays.shape[1:]),
                    dtype=tac_rgb_arrays.dtype,
                    compressor=compressor,
                    overwrite=True,
                )
                zarr_data.create_dataset(
                    "tac_force",
                    shape=(0, *tac_force_arrays.shape[1:]),
                    chunks=(batch_size, *tac_force_arrays.shape[1:]),
                    dtype=tac_force_arrays.dtype,
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
            zarr_data["camera"].append(camera_arrays)
            zarr_data["camera_pos"].append(camera_pos_arrays)
            zarr_data["tac_rgb"].append(tac_rgb_arrays)
            zarr_data["tac_pos"].append(tac_pos_arrays)
            zarr_data["tac_force"].append(tac_force_arrays)
            zarr_data["state"].append(state_arrays)
            zarr_data["action"].append(action_arrays)
            zarr_meta["episode_ends"].append(episode_ends_arrays)
            
            print(
                f"Batch {current_batch + 1} written with {len(camera_arrays)} samples."
            )
            
            print(f"camera shape: {camera_arrays.shape}")
            print(f"camera_pos shape: {camera_pos_arrays.shape}")
            print(f"tac_rgb shape: {tac_rgb_arrays.shape}")
            print(f"tac_force shape: {tac_force_arrays.shape}")
            print(f"tac_pos shape: {tac_pos_arrays.shape}")
            print(f"state shape: {state_arrays.shape}")
            print(f"action shape: {action_arrays.shape}")
            print(f"episode_ends shape: {episode_ends_arrays.shape}")
            
            # Clear arrays for next batch
            camera_arrays = []
            camera_pos_arrays = []
            tac_rgb_arrays = []
            tac_pos_arrays = []
            tac_force_arrays = []
            action_arrays = []
            state_arrays = []
            episode_ends_arrays = []
            current_batch += 1
            
if __name__ == "__main__":
    main()