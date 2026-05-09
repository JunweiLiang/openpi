"""
Script Json to Lerobot.

# --raw-dir     Corresponds to the directory of your JSON dataset
# --repo-id     Your unique repo ID on Hugging Face Hub
# --robot_type  The type of the robot used in the dataset (e.g., Unitree_G1_Dex3, Unitree_Z1_Dual, Unitree_G1_Dex3)
# --push_to_hub Whether or not to upload the dataset to Hugging Face Hub (true or false)

python unitree_lerobot/utils/convert_unitree_json_to_lerobot.py \
    --raw-dir $HOME/datasets/g1_grabcube_double_hand \
    --repo-id your_name/g1_grabcube_double_hand \
    --robot_type Unitree_G1_Dex3 \ 
    --push_to_hub
"""
import os
import cv2
import csv
import tqdm
import tyro
import json
import glob
import dataclasses
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Literal, List, Dict, Optional

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

@dataclasses.dataclass(frozen=True)
class RobotConfig:
    motors: List[str]
    cameras: List[str]
    camera_to_image_key:Dict[str, str]
    json_state_data_name: List[str]
    json_action_data_name: List[str]


Z1_CONFIG = RobotConfig(
    motors=[
        "kLeftWaist",
        "kLeftShoulder",
        "kLeftElbow",
        "kLeftForearmRoll",
        "kLeftWristAngle",
        "kLeftWristRotate",
        "kLeftGripper",
        "kRightWaist",
        "kRightShoulder",
        "kRightElbow",
        "kRightForearmRoll",
        "kRightWristAngle",
        "kRightWristRotate",
        "kRightGripper",
    ],
    cameras=[
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_high', 'color_1': 'cam_left_wrist' ,'color_2': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm'],
    json_action_data_name = ['left_arm', 'right_arm']
)


Z1_SINGLE_CONFIG = RobotConfig(
    motors=[
        "kWaist",
        "kShoulder",
        "kElbow",
        "kForearmRoll",
        "kWristAngle",
        "kWristRotate",
        "kGripper",
    ],
    cameras=[
        "cam_high",
        "cam_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_high', 'color_1': 'cam_wrist'},
    json_state_data_name = ['left_arm', 'right_arm'],
    json_action_data_name = ['left_arm', 'right_arm']
)


G1_GRIPPER_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftGripper",
        "kRightGripper",
    ],
    cameras=[
        "cam_left_high",
        # "cam_right_high",
        # "cam_left_wrist",
        # "cam_right_wrist",
        ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1':'cam_right_high', 'color_2': 'cam_left_wrist' ,'color_3': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
)


G1_DEX3_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftHandThumb0",
        "kLeftHandThumb1",
        "kLeftHandThumb2",
        "kLeftHandMiddle0",
        "kLeftHandMiddle1",
        "kLeftHandIndex0",
        "kLeftHandIndex1",
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ],
    cameras=[
        "cam_left_high",
        # "cam_right_high",
        # "cam_left_wrist",
        # "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_ee', 'right_ee'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_ee', 'right_ee']
)

G1_DEX3_RIGHT_ARM_CONFIG = RobotConfig(
    motors=[
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ],
    cameras=[
        "cam_left_high",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high'},
    json_state_data_name = ['right_arm', 'right_ee'],
    json_action_data_name = ['right_arm', 'right_ee']
)

G1_DEX3_RIGHT_ARM_GRIPPER_CAM2_CONFIG = RobotConfig(
    motors=[
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandIndex1",
    ],
    cameras=[
        "cam_left_high1",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1': 'cam_right_wrist'},
    json_state_data_name = ['right_arm', 'right_ee'],
    json_action_data_name = ['right_arm', 'right_ee']
)

G1_DEX3_RIGHT_ARM_TRIGGER_CAM2_CONFIG = RobotConfig(
    motors=[
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandTrigger",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_3': 'cam_right_wrist'},
    json_state_data_name = ['right_arm', 'right_ee'],
    json_action_data_name = ['right_arm', 'right_trigger']
)

G1_DEX3_DUAL_ARM_TRIGGER_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftHandTrigger",
        "kRightHandTrigger",
    ],
    cameras=[
        "cam_left_high",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_ee', 'right_ee'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_trigger', 'right_trigger']
)

G1_DEX3_DUAL_ARM_TRIGGER_CAM2_CONFIG = RobotConfig(
    motors=[
        "kLeftShoulderPitch",
        "kLeftShoulderRoll",
        "kLeftShoulderYaw",
        "kLeftElbow",
        "kLeftWristRoll",
        "kLeftWristPitch",
        "kLeftWristYaw",
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kLeftHandTrigger",
        "kRightHandTrigger",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_3': 'cam_right_wrist'},
    json_state_data_name = ['left_arm', 'right_arm', 'left_ee', 'right_ee'],
    json_action_data_name = ['left_arm', 'right_arm', 'left_trigger', 'right_trigger']
)

G1_DEX3_RIGHT_ARM_TRIGGER_WAIST_CAM2_CONFIG = RobotConfig(
    motors=[
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandTrigger",
        "kWaist",
    ],
    cameras=[
        "cam_left_high",
        'cam_right_wrist',
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_3': 'cam_right_wrist'},
    json_state_data_name = ['right_arm', 'right_ee', 'waist'],
    json_action_data_name = ['right_arm', 'right_trigger', 'waist']
)

G1_DEX3_RIGHT_ARM_GRIPPER_WAIST_CAM2_CONFIG = RobotConfig(
    motors=[
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandIndex1",
        "kWaist",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_wrist",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1': 'cam_right_wrist'},
    json_state_data_name = ['right_arm', 'right_ee', 'waist'],
    json_action_data_name = ['right_arm', 'right_ee', 'waist']
)

G1_DEX3_RIGHT_ARM_HEAD_DEPTH_CONFIG = RobotConfig(
    motors=[
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ],
    cameras=[
        "cam_left_high",
        "depth_left_high",
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'depth_0': 'depth_left_high'},
    json_state_data_name = ['right_arm', 'right_hand'],
    json_action_data_name = ['right_arm', 'right_hand']
)

G1_DEX3_RIGHT_ARM_HEAD_WRIST_DEPTH_CONFIG = RobotConfig(
    motors=[
        "kRightShoulderPitch",
        "kRightShoulderRoll",
        "kRightShoulderYaw",
        "kRightElbow",
        "kRightWristRoll",
        "kRightWristPitch",
        "kRightWristYaw",
        "kRightHandThumb0",
        "kRightHandThumb1",
        "kRightHandThumb2",
        "kRightHandIndex0",
        "kRightHandIndex1",
        "kRightHandMiddle0",
        "kRightHandMiddle1",
    ],
    cameras=[
        "cam_left_high",
        "cam_right_wrist",
        "depth_left_high",
        'depth_right_wrist'
    ],
    camera_to_image_key = {'color_0': 'cam_left_high', 'color_1': 'cam_right_wrist', 'depth_0': 'depth_left_high', 'depth_1': 'depth_right_wrist'},
    json_state_data_name = ['right_arm', 'right_hand'],
    json_action_data_name = ['right_arm', 'right_hand']
)


ROBOT_CONFIGS = {
    "Unitree_Z1_Single": Z1_SINGLE_CONFIG,
    "Unitree_Z1_Dual": Z1_CONFIG,
    "Unitree_G1_Gripper": G1_GRIPPER_CONFIG,
    "Unitree_G1_Dex3": G1_DEX3_CONFIG,
    "Unitree_G1_Dex3_Right_Arm": G1_DEX3_RIGHT_ARM_CONFIG,
    "Unitree_G1_Dex3_Right_Arm_Gripper_Cam2": G1_DEX3_RIGHT_ARM_GRIPPER_CAM2_CONFIG,
    "Unitree_G1_Dex3_Right_Arm_Trigger_Cam2": G1_DEX3_RIGHT_ARM_TRIGGER_CAM2_CONFIG,
    "Unitree_G1_Dex3_Dual_Arm_Trigger": G1_DEX3_DUAL_ARM_TRIGGER_CONFIG,
    "Unitree_G1_Dex3_Dual_Arm_Trigger_Cam2": G1_DEX3_DUAL_ARM_TRIGGER_CAM2_CONFIG,
    "Unitree_G1_Dex3_Right_Arm_Trigger_Waist_Cam2": G1_DEX3_RIGHT_ARM_TRIGGER_WAIST_CAM2_CONFIG,
    "Unitree_G1_Dex3_Right_Arm_Gripper_Waist_Cam2": G1_DEX3_RIGHT_ARM_GRIPPER_WAIST_CAM2_CONFIG,
    "Unitree_G1_Dex3_Right_Arm_Head_Depth": G1_DEX3_RIGHT_ARM_HEAD_DEPTH_CONFIG,
    "Unitree_G1_Dex3_Right_Arm_Head_Wrist_Depth": G1_DEX3_RIGHT_ARM_HEAD_WRIST_DEPTH_CONFIG,
}


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


class JsonDataset:
    def __init__(self, data_dirs: Path, robot_type: str) -> None:
        """
        Initialize the dataset for loading and processing HDF5 files containing robot manipulation data.
        
        Args:
            data_dirs: Path to directory containing training data
        """
        assert data_dirs is not None, "Data directory cannot be None"
        assert robot_type is not None, "Robot type cannot be None"
        self.data_dirs = data_dirs
        self.json_file = 'data.json'
        
        # Load task descriptions from CSV
        self.task_descriptions = self._load_task_descriptions()
        
        # Initialize paths and cache
        self._init_paths()
        self._init_cache()
        self.json_state_data_name = ROBOT_CONFIGS[robot_type].json_state_data_name
        self.json_action_data_name = ROBOT_CONFIGS[robot_type].json_action_data_name
        self.camera_to_image_key = ROBOT_CONFIGS[robot_type].camera_to_image_key
        
        # Detect format for each episode
        self.episode_formats = []
        for episode_path in self.episode_paths:
            self.episode_formats.append(self._detect_format(episode_path))
    
    def _load_task_descriptions(self) -> Dict[str, str]:
        """
        Load task descriptions from prompt.csv file.
        
        Returns:
            Dictionary mapping task names to their descriptions
        """
        task_map = {}
        csv_path = Path(__file__).parent / 'prompt.csv'
        
        if csv_path.exists():
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    if len(row) >= 3:
                        task_name = row[0]  # e.g., press_the_stapler_g1
                        description = row[2]  # the task description
                        task_map[task_name] = description
        
        return task_map
    
    def _detect_format(self, episode_path: str) -> str:
        """
        Detect whether the episode uses 'unloading' or 'press_the_stapler' format.
        
        Args:
            episode_path: Path to the episode directory
            
        Returns:
            'unloading' or 'press_the_stapler'
        """
        # Check if 'colors' directory exists (unloading format)
        colors_dir = os.path.join(episode_path, 'colors')
        color_dir = os.path.join(episode_path, 'color')
        
        if os.path.exists(colors_dir):
            return 'unloading'
        elif os.path.exists(color_dir):
            return 'press_the_stapler'
        else:
            # Default to unloading if uncertain
            return 'unloading'


    def _init_paths(self) -> None:
        """Initialize episode and task paths."""

        self.episode_paths = []
        self.task_paths = []
        
        for task_path in glob.glob(os.path.join(self.data_dirs, '*')):
            if os.path.isdir(task_path):
                # Only collect directories that start with 'episode'
                episode_paths = [
                    ep for ep in glob.glob(os.path.join(task_path, '*'))
                    if os.path.isdir(ep) and os.path.basename(ep).startswith('episode')
                ]
                if episode_paths:
                    self.task_paths.append(task_path)
                    self.episode_paths.extend(episode_paths)
        
        self.episode_paths = sorted(self.episode_paths)
        self.episode_ids = list(range(len(self.episode_paths)))


    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self.episode_paths)


    def _init_cache(self) -> List:
        """Initialize data cache if enabled."""

        self.episodes_data_cached = []
        for episode_path in tqdm.tqdm(self.episode_paths, desc="Loading Cache Json"):
            json_path = os.path.join(episode_path, self.json_file)
            print(f"Loading json file: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as jsonf:
                self.episodes_data_cached.append(json.load(jsonf))

        print(f"==> Cached {len(self.episodes_data_cached)} episodes")

        return self.episodes_data_cached


    def _extract_data(self, episode_data: Dict, key: str, parts: List[str], data_format: str) -> np.ndarray:
        """
        Extract data from episode dictionary for specified parts.
        
        Args:
            episode_data: Dictionary containing episode data
            key: Data key to extract ('states' or 'actions')
            parts: List of parts to include ('left_arm', 'right_arm')
            data_format: 'unloading' or 'press_the_stapler'
            
        Returns:
            Concatenated numpy array of the requested data
        """
        result = []
        
        if data_format == 'unloading':
            # Original unloading format
            for sample_data in episode_data['data']:
                data_array = np.array([], dtype=np.float32)
                for part in parts:
                    if part in sample_data[key] and sample_data[key][part] is not None:
                        if part == "right_trigger" or part == "left_trigger":
                            qpos = np.array([sample_data[key][part]], dtype=np.float32)
                        else:
                            qpos = np.array(sample_data[key][part]['qpos'], dtype=np.float32)
                        # 用第4个关节来控制夹爪
                        if part == 'left_hand' or part == 'right_hand':
                            qpos = np.array([qpos[4]])  # Only take the 4th joint
                        if part == 'left_ee' or part == 'right_ee': # 用第4个关节来控制夹爪
                            qpos = np.array([qpos[4]])
                        if part == 'waist': # 用第1个关节来控制腰部
                            qpos = np.array([qpos[0]])
                        data_array = np.concatenate([data_array, qpos])
                result.append(data_array)
        else:
            # press_the_stapler format
            for sample_data in episode_data:
                data_array = np.array([], dtype=np.float32)
                for part in parts:
                    if part == 'left_arm':
                        # states.arm_state or actions.sol_q, take last 7 elements
                        if key == 'states':
                            qpos = np.array(sample_data[key]['arm_state'][:7], dtype=np.float32)
                        else:  # actions
                            qpos = np.array(sample_data[key]['sol_q'][:7], dtype=np.float32)
                    elif part == 'right_arm':
                        # states.arm_state or actions.sol_q, take last 7 elements
                        if key == 'states':
                            qpos = np.array(sample_data[key]['arm_state'][-7:], dtype=np.float32)
                        else:  # actions
                            qpos = np.array(sample_data[key]['sol_q'][-7:], dtype=np.float32)
                    elif part == 'left_ee' or part == 'left_trigger':
                        # states.hand_state or actions.right_angles
                        if key == 'states':
                            # qpos = np.array(sample_data[key]['hand_state'][:7], dtype=np.float32)
                            qpos = np.array([sample_data[key]['hand_state'][4]], dtype=np.float32)
                        else:  # actions
                            # qpos = np.array(sample_data[key]['left_angles'], dtype=np.float32)
                            qpos = np.array([sample_data[key]['left_angles'][4]], dtype=np.float32)
                    elif part == 'right_ee' or part == 'right_trigger':
                        # states.hand_state or actions.right_angles
                        if key == 'states':
                            # qpos = np.array(sample_data[key]['hand_state'][-7:], dtype=np.float32)
                            qpos = np.array([sample_data[key]['hand_state'][-3]], dtype=np.float32)
                        else:  # actions
                            # qpos = np.array(sample_data[key]['right_angles'], dtype=np.float32)
                            qpos = np.array([sample_data[key]['right_angles'][4]], dtype=np.float32)
                    else:
                        continue
                    data_array = np.concatenate([data_array, qpos])
                result.append(data_array)
        
        return np.array(result)


    def _parse_images(self, episode_path: str, episode_data, data_format: str) -> dict[str, list[np.ndarray]]:
        """Load and stack images for a given camera key."""

        images = defaultdict(list)

        if data_format == 'unloading':
            # Original unloading format
            keys = list(episode_data["data"][0]['colors'].keys()) + list(episode_data["data"][0].get('depths', {}).keys())
            cameras = [key for key in keys]

            for camera in cameras:
                image_key = self.camera_to_image_key.get(camera)
                if image_key is None:
                    continue

                for sample_data in episode_data['data']:
                    # 判断是 color 还是 depth 图像
                    if camera in sample_data.get('colors', {}):
                        relative_path = sample_data['colors'].get(camera)
                        if not relative_path:
                            continue
                        image_path = os.path.join(episode_path, relative_path)
                        if not os.path.exists(image_path):
                            raise FileNotFoundError(f"Image path does not exist: {image_path}")
                        image = cv2.imread(image_path)
                        if image is None:
                            raise RuntimeError(f"Failed to read image: {image_path}")
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images[image_key].append(image_rgb)
                    elif camera in sample_data.get('depths', {}):
                        relative_path = sample_data['depths'].get(camera)
                        if not relative_path:
                            continue
                        img_path = os.path.join(episode_path, relative_path)
                        if not os.path.exists(img_path):
                            raise FileNotFoundError(f"Depth image path does not exist: {img_path}")
                        depth_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                        if depth_img is None:
                            raise RuntimeError(f"Failed to read depth image: {img_path}")
                        # 归一化到8位
                        depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
                        depth_normalized = depth_normalized.astype('uint8')
                        # 伪彩色
                        colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        images[image_key].append(colored_depth)
        else:
            # press_the_stapler format
            # Images are in color/ and depth/ directories with frame_XXXXXX.jpg naming
            color_dir = os.path.join(episode_path, 'color')
            depth_dir = os.path.join(episode_path, 'depth')
            
            # Get sorted list of color images
            if os.path.exists(color_dir):
                color_files = sorted([f for f in os.listdir(color_dir) if f.startswith('frame_') and f.endswith('.jpg')])
                image_key = self.camera_to_image_key.get('color_0')
                if image_key:
                    for color_file in color_files:
                        image_path = os.path.join(color_dir, color_file)
                        image = cv2.imread(image_path)
                        if image is None:
                            raise RuntimeError(f"Failed to read image: {image_path}")
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images[image_key].append(image_rgb)
            
            # Get sorted list of depth images if needed
            if os.path.exists(depth_dir):
                depth_files = sorted([f for f in os.listdir(depth_dir) if f.startswith('frame_') and f.endswith('.jpg')])
                depth_key = self.camera_to_image_key.get('depth_0')
                if depth_key:
                    for depth_file in depth_files:
                        img_path = os.path.join(depth_dir, depth_file)
                        depth_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                        if depth_img is None:
                            raise RuntimeError(f"Failed to read depth image: {img_path}")
                        # 归一化到8位
                        depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
                        depth_normalized = depth_normalized.astype('uint8')
                        # 伪彩色
                        colored_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        images[depth_key].append(colored_depth)

        return images


    def get_item(self, index: Optional[int] = None,) -> Dict:
        """Get a training sample from the dataset.  """
            
        file_path = np.random.choice(self.episode_paths) if index is None else self.episode_paths[index]
        episode_data = self.episodes_data_cached[index]
        data_format = self.episode_formats[index]

        # Load state and action data
        action = self._extract_data(episode_data, 'actions', self.json_action_data_name, data_format)
        state = self._extract_data(episode_data, 'states', self.json_state_data_name, data_format)
        episode_length = len(state)
        state_dim = state.shape[1] if len(state.shape) == 2 else state.shape[0]
        action_dim = action.shape[1] if len(action.shape) == 2 else state.shape[0]
        
        # Load task description
        if data_format == 'unloading':
            task = episode_data.get('text', {}).get('goal', "")
        else:
            # Look up task description from CSV based on folder name
            task_folder = os.path.basename(os.path.dirname(file_path))
            # Try with _g1 suffix first (e.g., press_the_stapler -> press_the_stapler_g1)
            task_key_with_suffix = f"{task_folder}_g1"
            task = self.task_descriptions.get(task_key_with_suffix, "")
            # If not found, try without suffix
            if not task:
                task = self.task_descriptions.get(task_folder, "")
            # If still not found, fallback to replacing underscores with spaces
            if not task:
                task = task_folder.replace('_', ' ')
        
        # Load camera images
        cameras = self._parse_images(file_path, episode_data, data_format)

        # Extract camera configuration
        cam_height, cam_width = next(img for imgs in cameras.values() if imgs for img in imgs).shape[:2]
        data_cfg = {
            'camera_names': list(cameras.keys()),
            'cam_height': cam_height,
            'cam_width': cam_width,
            'state_dim': state_dim,
            'action_dim': action_dim,
        }
        
        return {'episode_index': index,
                'episode_length': episode_length,
                'state': state, 
                'action': action,
                'cameras': cameras,
                'task': task,
                'data_cfg':data_cfg,
                'episode_path': file_path}


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    
    motors = ROBOT_CONFIGS[robot_type].motors
    cameras = ROBOT_CONFIGS[robot_type].cameras

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    raw_dir: Path,
    robot_type: str,
) -> LeRobotDataset:

    json_dataset = JsonDataset(raw_dir, robot_type)
    for i in tqdm.tqdm(range(len(json_dataset))):
        episode = json_dataset.get_item(i)

        state = episode["state"]
        action = episode["action"]
        cameras = episode["cameras"]
        task = episode["task"]
        episode_length = episode["episode_length"]
        episode_path = episode["episode_path"]
        
        print(f"\n[Episode {i+1}/{len(json_dataset)}] Processing: {episode_path}")
        print(f"  Frames: {episode_length}, State dim: {state.shape}, Action dim: {action.shape}")
        episode_path = episode["episode_path"]
        
        print(f"\n[{i+1}/{len(json_dataset)}] Processing episode: {episode_path}")
        
        # Skip episodes with no data
        if episode_length == 0:
            print(f"Warning: Skipping episode {i} - no frames")
            continue
        
        # Validate that all cameras have the same number of frames
        if cameras:
            camera_lengths = {cam: len(imgs) for cam, imgs in cameras.items()}
            if len(set(camera_lengths.values())) > 1:
                print(f"Warning: Skipping episode {i} - inconsistent camera frame counts: {camera_lengths}")
                continue
            # Check if camera frames match state/action length
            first_cam_length = next(iter(camera_lengths.values()))
            if first_cam_length != episode_length:
                print(f"Warning: Episode {i} - camera frames ({first_cam_length}) != state/action length ({episode_length})")
                print(f"  Using minimum length")
                episode_length = min(episode_length, first_cam_length)

        num_frames = episode_length
        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": task
            }

            for camera, img_array in cameras.items():
                img_resized = cv2.resize(img_array[i], (640, 480))
                img_resized = np.transpose(img_resized, (2, 0, 1))
                frame[f"observation.images.{camera}"] = img_resized

            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def json_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,        # Unitree_Z1_Dual, Unitree_G1_Gripper, Unitree_G1_Dex3
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):

    if (HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        mode=mode,
        has_effort=False,
        has_velocity=False,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        raw_dir,
        robot_type=robot_type,
    )

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder = True)


def local_push_to_hub(
        repo_id: str,
        root_path: Path,):

    dataset = LeRobotDataset(repo_id = repo_id, root = root_path)
    dataset.push_to_hub(upload_large_folder = True)


if __name__ == "__main__":
    tyro.cli(json_to_lerobot)
