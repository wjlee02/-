import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython

import numpy as np
from ultralytics import YOLO
import cv2

import json

import readline
from sensor_msgs.msg import CompressedImage


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, task_space, vel_control):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.task_space = task_space
        self.vel_control = vel_control
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            # is_sim = root.attrs['sim']
            is_sim = None
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            qpos = root['/observations/qpos'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                action = root['/action'][max(0, start_ts - 1):]
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes, task_space=False, vel_control=False):
    all_qpos_data = []
    all_action_data = []
    # episode 길이 관련 코드 수정
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            if task_space:
                if vel_control:
                    # qpos = root['/observations/xvel'][()]
                    qpos = np.zeros_like(root['/observations/xvel'][()])
                    action = root['/xvel_action'][()]
                else:
                    qpos = root['/observations/xpos'][()]
                    action = root['/xaction'][()]
            else:
                qpos = root['/observations/qpos'][()]
                action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}
    
    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, task_space, vel_control):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes, task_space, vel_control)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, task_space, vel_control)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, task_space, vel_control)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def zoom_image(img, point, size):
    height, width = img.shape[:2]

    
    # 중심 좌표를 기준으로 관심 영역 크기 계산
    half_width = int(size[0] / 2)
    half_height = int(size[1] / 2)

    x1 = max(point[0] - half_width, 0)
    y1 = max(point[1] - half_height, 0)
    x2 = min(point[0] + half_width, width)
    y2 = min(point[1] + half_height, height)

    # 관심 영역 크롭
    cropped = img[y1:y2, x1:x2]
    
    return cropped



def fetch_image_with_config(image, config, memory=None, yolo_config=None, no_yolo=False, no_zoom=False):
    if memory is None:
        memory = {
            'is_first_image': True,
            'fixed_boxes': [],
            'last_box': {}
        }

    if 'zoom' in config and not no_zoom:
        size = config['zoom']['size']
        point = config['zoom']['point']
        image = zoom_image(image, point, size)
    if 'resize' in config:
        size = config['resize']['size']
        image = cv2.resize(image, size)
    if 'masked_yolo' in config and not no_yolo:
        classes = config['masked_yolo']['classes']
        masked_image = np.zeros_like(image)
        show_boxes = []
        yolo_model = yolo_config['model']

        results = yolo_model(image, conf=yolo_config['conf'])
        result = results[0]
        boxes = result.boxes
        names = result.names
        masked_image = np.zeros_like(image)

        for class_name, class_config in classes.items():
            class_boxes = []
            last_box = memory['last_box'][class_name] if class_name in memory['last_box'] else None
            for box in boxes:
                box_id = int(box.cls.item())
                if box_id == class_config['id']:
                    class_boxes.append(box)

            if class_config['is_fixed_mask']:

                if memory['is_first_image']:

                    if class_config['show_id'] == -1:
                        memory['fixed_boxes'] += class_boxes
                        memory['is_first_image'] = False
                    elif len(class_boxes) > class_config['show_id']:
                        memory['fixed_boxes'].append(class_boxes[class_config['show_id']])
                        memory['is_first_image'] = False

                show_boxes += memory['fixed_boxes']

            else:
                if class_config['show_id'] == -1:
                    show_boxes += class_boxes
                elif len(class_boxes) > class_config['show_id']:
                    show_boxes.append(class_boxes[class_config['show_id']])

            if class_config['keep_last_box']:
                if last_box is not None:
                    show_boxes.append(last_box)
                if len(class_boxes) > 0:
                    memory['last_box'][class_name] = class_boxes[0]

        if len(show_boxes) > 0:
            masked_image = mask_outside_boxes(image, show_boxes, padding=yolo_config['padding'])
        image = masked_image

    
    return image, memory


def mask_outside_boxes(image, boxes_list, padding=0):
    """
    박스 내부 이미지만 살리고, 나머지 영역은 검게 칠하는 함수.
    """

    height, width, _ = image.shape
    # 원본 이미지와 동일한 크기의 흰색 이미지 초기화
    masked_image = np.full_like(image, 255)

    for boxes in boxes_list:

        # YOLO 박스에서 xyxy 좌표 가져오기 (tensor 형태)
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4) 형태의 NumPy 배열로 변환

        # 박스별로 반복하며 박스 영역을 복사
        for box in xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)

            masked_image[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    return masked_image


def input_caching(prompt):
    cache_file_path = "input_cache.json"
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}
    default = cache.get(prompt, "")
    def prefill_hook():
        readline.insert_text(default)  # 기본값 입력
        readline.redisplay()          # 화면에 표시
    readline.set_pre_input_hook(prefill_hook)

    answer = input(prompt)

    cache[prompt] = answer

    with open(cache_file_path, "w") as f:
        json.dump(cache, f, indent=4)

    return answer


def qpos_to_xpos(qpos, kn, robot='ur5e'):
    qpos_only = list(qpos[:-1])
    if robot == 'ur5e':
        qpos_only[0], qpos_only[2] = qpos_only[2], qpos_only[0]
    xpos = kn.forward_kinematics(qpos_only).detach().cpu().numpy()
    return np.concatenate((xpos, qpos[-1:]))

def xpos_to_qpos(xpos, kn, qseed, robot='ur5e'):
    xpos_only = list(xpos[:-1])
    qpos = kn.inverse_kinematics(xpos_only, qseed).detach().cpu().numpy()
    if robot == 'ur5e':
        qpos[0], qpos[2] = qpos[2], qpos[0]
    return np.concatenate((qpos, xpos[-1:]))


def ros_image_to_numpy(image_msg):
    if isinstance(image_msg, CompressedImage):
        # 압축 이미지 처리
        np_arr = np.frombuffer(image_msg.data, np.uint8)
        image_array = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # 기본 BGR 형태로 디코딩됨
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)  # RGB로 변환
        image_array = image_array[:, :, ::-1]  # BGR -> RGB
        return image_array

    # 일반 Image 메시지 처리
    encoding_to_dtype = {
        'rgb8': ('uint8', 3),
        'bgr8': ('uint8', 3),
        'mono8': ('uint8', 1),
        'mono16': ('uint16', 1),
        'rgba8': ('uint8', 4),
        'bgra8': ('uint8', 4),
    }

    if image_msg.encoding not in encoding_to_dtype:
        raise ValueError(f"Unsupported encoding: {image_msg.encoding}")
    
    dtype, channels = encoding_to_dtype[image_msg.encoding]
    data = np.frombuffer(image_msg.data, dtype=dtype)
    image_array = data.reshape((image_msg.height, image_msg.width, channels))
    
    if image_msg.encoding == 'bgr8':
        image_array = image_array[:, :, ::-1]  # BGR -> RGB
    elif image_msg.encoding == 'bgra8':
        image_array = image_array[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA
    
    return image_array

def rescale_val(val, origin_rng, rescaled_rng):
    return rescaled_rng[0] + (rescaled_rng[1] - rescaled_rng[0]) * ((val - origin_rng[0]) / (origin_rng[1] - origin_rng[0]))