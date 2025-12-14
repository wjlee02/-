import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import time
from matplotlib.animation import FuncAnimation

from constants import DT
from constants import TASK_CONFIGS
from policy import ACTPolicy, CNNMLPPolicy
from utils import load_data, zoom_image, fetch_image_with_config # data functions
from utils import sample_box_pose, sample_insertion_pose, qpos_to_xpos, xpos_to_qpos # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from visualize_episodes import save_videos

# from apply_yolo import mask_outside_boxes

import shutil
import random
import h5py

import cv2

# from TCPController import TCPController


from pynput import keyboard

import sys

IMG_SIZE = (120, 90)


def main(args):
    set_seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device=="cuda": torch.cuda.empty_cache()
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    record_episode = args['record_episode']
    data_folders = args['data_folders']
    load_model = args['load_model']
    task_space = args['task_space']
    vel_control = args['vel_control']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    camera_config = task_config['camera_config']
    home_pose = task_config['home_pose']
    end_pose = task_config['end_pose']
    pose_sleep = task_config['pose_sleep']

    # fixed parameters
    state_dim = 8 if task_space else 14
    lr_backbone = 1e-5
    backbone = 'resnet18' if args['backbone'] is None else args['backbone']
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'camera_config': camera_config,
        'real_robot': not is_sim,
        'end_pose': end_pose,
        'home_pose': home_pose,
        'pose_sleep': pose_sleep,
        'data_folders': data_folders,
        'load_model': load_model,
        'dataset_dir': dataset_dir,
        'task_space': task_space,
        'vel_control': vel_control
    }

    # evaluation
    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, record_episode=record_episode)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    start_num = 0

    for i, folder in enumerate(config['data_folders']):
        if i % 2 == 0:
            sampling = int(config['data_folders'][i + 1])
            origin_dir = f"{dataset_dir}/{folder}"
            if os.path.exists(origin_dir):
                file_count = move_data_to_tmp(f"{origin_dir}", f"{dataset_dir}/tmp", start_num, sampling)
                start_num += file_count
            else:
                print(f"{origin_dir} 폴더가 존재하지 하지 않습니다.")

    train_dataloader, val_dataloader, stats, _ = load_data(f"{dataset_dir}/tmp", start_num, camera_names, batch_size_train, batch_size_val, task_space, vel_control)



    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)

    tmp_folder_path = f"{dataset_dir}/tmp"
    if os.path.exists(tmp_folder_path):
        shutil.rmtree(tmp_folder_path)
        print(f"'{tmp_folder_path}' 폴더가 삭제되었습니다.")
    else:
        print(f"'{tmp_folder_path}' 폴더가 존재하지 않습니다.")

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')

def get_image(ts, camera_names, camera_config, memories):
    curr_images = []
    raw_images = []

    for index, cam_name in enumerate(camera_names):
        image = ts.observation['images'][cam_name]

        if cam_name in camera_config:
            image, memories[index] = fetch_image_with_config(image, camera_config[cam_name], memories[index], no_yolo=True)
        
        image = cv2.resize(image, IMG_SIZE)
        # print(image.shape)

        raw_images.append(image)
        curr_image = rearrange(image, 'h w c -> c h w')
        curr_images.append(curr_image)

    if len(raw_images):
        # 이미지 크기 맞추기 (최대 크기로 맞추거나 다른 방식으로 조정)
        max_height = 90
        max_width = 120
        resized_images = [cv2.resize(img, (max_width, max_height)) for img in raw_images]

        # 이미지를 가로로 나열
        combined_image = cv2.hconcat(resized_images)

        # 단일 창에 표시
        cv2.imshow("Combined Image", combined_image)
        cv2.waitKey(1)
    else:
        print("No images to display.")

    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    return curr_image, memories

def eval_bc(config, ckpt_name, record_episode=True):

    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    dataset_dir = config['dataset_dir']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    home_pose = config['home_pose']
    end_pose = config['end_pose']
    pose_sleep = config['pose_sleep']
    task_space = config['task_space']
    vel_control = config['vel_control']

    dataset_dir = f'{dataset_dir}/original'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']


    # load environment
    # if real_robot:
    # from aloha_scripts.robot_utils import move_grippers # requires aloha
    from env import AlohaEnv
    import rclpy

    rclpy.init()
    env = AlohaEnv(camera_names)
    env_max_reward = 0
    for _ in range(20):
        rclpy.spin_once(env, timeout_sec=0.01)

    # else:
        # from sim_env import make_sim_env
        # env = make_sim_env(task_name)
        # env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    num_queries = 0
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    data_timesteps = max_timesteps
    max_timesteps = int(max_timesteps * 2) # may increase for real-world tasks

    num_rollouts = 100
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):

        rollout_id += 0

        # if 'sim_transfer_cube' in task_name:
        #     BOX_POSE[0] = sample_box_pose() # used in sim reset
        # elif 'sim_insertion' in task_name:
        #     BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        # for i in range(3):
        print('go home pose!')
        env.move_step(home_pose)
        time.sleep(pose_sleep)

        ts = env.reset()
        timesteps = [ts]
        actions = []
        xactions = []
        
        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda() 

        rewards = []

        if onscreen_render:
            plt.ion()  # interactive mode on
            fig, ax = plt.subplots()


        print('move!')
        memories = [None] * len(camera_names)

        ts = env.reset()
        timesteps = [ts]
        actions = []
        
        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        rewards = []
        print("Start")
        for t in tqdm(range(max_timesteps)):
            for _ in range(20):
                rclpy.spin_once(env, timeout_sec=0.01)
            # try:
            start = time.time()
            ### process previous timestep to get qpos and image_list
            timesteps.append(ts)
            obs = ts.observation
                
            cur_qpos = np.array(obs['qpos'])
            cur_qvel = np.array(obs['qvel'])


            robot_input_raw = cur_qpos
            
            # pos_numpy = np.array(obs['xpos']) if task_space else np.array(obs['qpos'])
            
            robot_input = pre_process(robot_input_raw)
            robot_input = torch.from_numpy(robot_input).float().cuda().unsqueeze(0)

            curr_image, memories = get_image(ts, camera_names, config['camera_config'], memories)
            
            with torch.inference_mode():
                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(robot_input, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        # raw_action = all_actions[:, t % query_frequency]
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(robot_input, curr_image)
                else:
                    raise NotImplementedError

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action)


            target_qpos = action
            # print(target_qpos)
            ts = env.move_step(target_qpos)

            rewards.append(ts.reward)

            actions.append(target_qpos)

            
        plt.close()
        if real_robot:
            pass

        env.move_step(end_pose)
        time.sleep(pose_sleep)
            
        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        if record_episode:
            d = input("Will you save this data? (Input 'y' for yes)")
            if len(d) > 0 and d[-1] == 'y':
                record_eval_episode(timesteps, actions, xactions, camera_names, config['camera_config'], data_timesteps, state_dim, dataset_dir, yolo_config, kin_config)
            else:
                d = input("Will you record new data? (Input 'y' for yes)")
                if d == 'y':
                    record_new_episode(task_name)


    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def record_eval_episode(timesteps, actions, xactions, camera_names, camera_config, max_timesteps, joint_len, dataset_dir, yolo_config, kin_config):
    timesteps.pop(len(timesteps) - 1)

    image_size = (720, 1280)
    
    while len(timesteps) >= max_timesteps:
        data_dict = {
            '/observations/qpos': [],
            '/observations/xpos': [],
            '/observations/xvel': [],
            '/observations/qvel': [],
            '/observations/effort': [],
            '/action': [],
        }
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []
            
        episode_i = get_auto_index(dataset_dir=dataset_dir)
        memory = None
        for j in range(max_timesteps):
            ts = timesteps.pop(0)
            action = actions.pop(0)
            xaction = xactions.pop(0)
            if j == 0:
                last_xpos = ts.observation['xpos']
                last_xaction = xaction
            
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            xpos = ts.observation['xpos']
            data_dict['/observations/xpos'].append(xpos)
            data_dict['/observations/effort'].append(ts.observation['effort'])
            data_dict['/action'].append(action)
            data_dict['/xaction'].append(xaction)
            
            xvel = last_xpos - xpos
            data_dict['/observations/xvel'].append(xvel)
            
            xvel_action = last_xaction - xaction
            data_dict['/xvel_action'].append(xvel_action)
            
            last_xpos = xpos
            last_xaction = xaction
            

            for cam_name in camera_names:
                image = ts.observation['images'][cam_name]

                if cam_name in camera_config:

                    image, memory = fetch_image_with_config(image, camera_config[cam_name], memory, yolo_config)

                if ts.observation['images'][cam_name] is not None:
                    data_dict[f'/observations/images/{cam_name}'].append(image)
                else:
                    print("error")

            # HDF5
        t0 = time.time()
        dataset_name = dataset_dir + '/episode_' + str(episode_i) + '.hdf5'
        with h5py.File(dataset_name, 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, image_size[0], image_size[1], 3), dtype='uint8',
                                        chunks=(1, image_size[0], image_size[1], 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            _ = obs.create_dataset('qpos', (max_timesteps, 7))
            _ = obs.create_dataset('xpos', (max_timesteps, 8))
            _ = obs.create_dataset('xvel', (max_timesteps, 8))
            _ = obs.create_dataset('qvel', (max_timesteps, 7))
            _ = obs.create_dataset('effort', (max_timesteps, 7))
            _ = root.create_dataset('action', (max_timesteps, 7))
            _ = root.create_dataset('xaction', (max_timesteps, 8))
            _ = root.create_dataset('xvel_action', (max_timesteps, 8))
            

            for name, array in data_dict.items():
                root[name][...] = array

        print(f'Saving: {time.time() - t0:.1f} secs', dataset_name)


def record_new_episode(task_name):
    import subprocess

    cmd1 = ['python', 'dynamixel_publisher.py']
    cmd2 = ["bash", "auto_record.sh", task_name, '2']

    p1 = subprocess.Popen(cmd1)
    p2 = subprocess.Popen(cmd2)

    try:
        while p2.poll() is None:  # p2가 실행 중인지 확인
            time.sleep(1)
    finally:
        print("Record Episode has Finished.")
        p1.terminate()  # p2가 종료되면 p1도 종료
        p1.wait()  # 완전히 종료될 때까지 기다리기
        print("Both process has finished")

    return




def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    load_model = config['load_model']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    
    # 모델 불러와서 fine-tuning
    if load_model != None:
        model_ckpt_name = 'policy_best.ckpt'
        model_path = os.path.join(load_model, model_ckpt_name)
        loading_status = policy.load_state_dict(torch.load(model_path))
        print(loading_status)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # if epoch % 100 == 0:
        #     ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
        #     torch.save(policy.state_dict(), ckpt_path)
        #     plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    # ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    # torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    # plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


import os
import shutil
import random

def move_data_to_tmp(origin_dir, tmp_dir, data_len, sampling):
    # 대상 디렉토리가 없으면 생성
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    # origin_dir 내의 파일 목록 (파일만 선택)
    file_list = [f for f in os.listdir(origin_dir) if os.path.isfile(os.path.join(origin_dir, f))]
    n = len(file_list)
    
    # sampling이 -1이면 모든 파일 복사
    if sampling == -1:
        selected_files = file_list
    # sampling이 파일 개수 이하이면 중복 없이 샘플링
    elif sampling <= n:
        selected_files = random.sample(file_list, sampling)
    else:
        # sampling 값이 파일 개수보다 크면 중복 허용
        # 각 파일이 몇 번까지 복사될 수 있는지 계산:
        # 예를 들어, n < sampling <= 2*n이면 각 파일은 최대 2번,
        # 2*n < sampling <= 3*n이면 최대 3번 등
        max_repeat = (sampling - 1) // n + 1
        
        # 각 파일을 max_repeat번씩 복사해서 확장된 리스트 생성
        extended_list = []
        for file in file_list:
            extended_list.extend([file] * max_repeat)
        
        # 확장된 리스트에서 sampling 개수만큼 랜덤 샘플링 (중복 가능)
        selected_files = random.sample(extended_list, sampling)
    
    file_count = 0
    for i, file_name in enumerate(selected_files):
        file_path = os.path.join(origin_dir, file_name)  # 원본 파일 경로
        new_file_name = f"episode_{data_len + i}.hdf5"     # 새로운 파일 이름 생성
        new_file_path = os.path.join(tmp_dir, new_file_name) # 대상 경로
        
        shutil.copy(file_path, new_file_path)  # 파일 복사
        print(f"{origin_dir}: {file_name} → {new_file_name} 로 복사 완료.")
        file_count += 1

    return file_count


def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    # parser.add_argument('--speed', action='store', type=int, help='speed', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--record_episode', action='store_true')
    parser.add_argument('--data_folders', nargs='+', type=str, required=False)
    parser.add_argument('--load_model', action='store', type=str, required=False)
    parser.add_argument('--task_space', action='store_true')
    parser.add_argument('--vel_control', action='store_true')
    
    main(vars(parser.parse_args()))

    
