import rclpy
from rclpy.node import Node
import threading
import math
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, CompressedImage
import time
import dm_env
import collections
from cv_bridge import CvBridge
from constants import DT, DEFAULT_CAMERA_NAMES, JOINT_LIMIT, TOPIC_NAME, JOINT_NAMES, TOOL_NAMES
from sensor_msgs.msg import JointState
from PIL import Image as PILImage
import numpy as np
import sys
from utils import qpos_to_xpos, xpos_to_qpos, ros_image_to_numpy, rescale_val
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from functools import partial


class AlohaEnv(Node):
    def __init__(self, camera_names=DEFAULT_CAMERA_NAMES, robot_name="piper"):
        super().__init__('aloha_env')
        self.robot_name = robot_name
        self.left_joint_states = None
        self.right_joint_states = None
        self.master_left_joint_states = None
        self.master_right_joint_states = None
        self.camera_names = camera_names
        self.js_mutex = threading.Lock()
        self.bridge = CvBridge()
        self.is_showing = False
        self.joint_names = JOINT_NAMES[robot_name]
        self.pyversion = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        if robot_name == 'piper':
            self.create_subscription(JointState, '/left/joint_states_single', self.left_joint_state_cb, qos_profile_sensor_data)
            self.create_subscription(JointState, '/right/joint_states_single', self.right_joint_state_cb, qos_profile_sensor_data)
            self.create_subscription(JointState, '/left/joint_states', self.master_left_joint_state_cb, qos_profile_sensor_data)
            self.create_subscription(JointState, '/right/joint_states', self.master_right_joint_state_cb, qos_profile_sensor_data)
            self.right_joint_pub = self.create_publisher(JointState, '/right/joint_states', 10)
            self.left_joint_pub = self.create_publisher(JointState, '/left/joint_states', 10)

        for cam_name in camera_names:
            self.create_subscription(CompressedImage, f"/{cam_name}/{cam_name}/color/image_raw/compressed", partial(self.image_raw_cb, cam_name), qos_profile_sensor_data)

        time.sleep(0.1)


    def left_joint_state_cb(self, msg):
        with self.js_mutex:
            self.left_joint_states = msg

    def right_joint_state_cb(self, msg):
        with self.js_mutex:
            self.right_joint_states = msg

    def master_left_joint_state_cb(self, msg):
        with self.js_mutex:
            self.master_left_joint_states = msg

    def master_right_joint_state_cb(self, msg):
        with self.js_mutex:
            self.master_right_joint_states = msg
    
    def image_raw_cb(self, cam_name, msg):
        img = ros_image_to_numpy(msg)
        setattr(self, cam_name, img)

    def get_reward(self):
        return 0
    
    def get_observation(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        return obs
    
    def reset(self):
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())
    
    def record_step(self):
        time.sleep(DT)
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation())
    

    def fetch_action(self, action):
        joint_limit = [
            (-2.618, 2.168),
            (0, 2.2),
            (-2.967, 0),
            (-1.745, 1.745),
            (-1.220, 1.220),
            (-2.094, 2.094),
            (0, 0.4),

            (-2.618, 2.168),
            (0, 2.2),
            (-2.967, 0),
            (-1.745, 1.745),
            (-1.220, 1.220),
            (-2.094, 2.094),
            (0, 0.4)
        ]
        gripper_scale = [0.029, 0.0299, 0.35]
        for i, a in enumerate(action):
            min = joint_limit[i][0]
            max = joint_limit[i][1]
            
            if min > a:
                a = min + 0.01
            elif max < a:
                a = max - 0.01


            if i == 6 or i == 13:
                for slave_pos in gripper_scale:
                    if a < slave_pos + 0.0003:
                        a = slave_pos
                        break
            
            action[i] = float(a)
        
        return action
        
    def move_step(self, action):
        """
        imitate_episode.py
        replay_eposode.py
        
        """
        action = self.fetch_action(action)
        js = JointState()
        js.header = Header()
        js.name = self.joint_names
        js.header.stamp = Time(sec=0, nanosec=0)
        js.header.frame_id = ''
        js.position = [float(x) for x in action[:7]]

        # js.position = action[:7]
        self.right_joint_pub.publish(js)
        
        js.position = [float(x) for x in action[-7:]]
        # js.position = action[-7:]
        self.left_joint_pub.publish(js)

        print(js)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation()
        )


    def get_qpos(self):
        return self.right_joint_states.position + self.left_joint_states.position

    def get_qvel(self):
        return self.right_joint_states.velocity + self.left_joint_states.velocity
    
    def get_effort(self):
        return self.right_joint_states.effort + self.left_joint_states.effort

    def get_action(self):
        return self.master_right_joint_states.position + self.master_left_joint_states.position

    def get_images(self):
        image_dict = dict()
        for cam_name in self.camera_names:
            image_dict[cam_name] = getattr(self, f'{cam_name}')
        return image_dict
    
        
def make_env(camera_names):
    env = AlohaEnv(camera_names)
    return env



