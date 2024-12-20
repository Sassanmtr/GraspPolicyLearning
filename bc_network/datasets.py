import os
import fnmatch
import random
import numpy as np
from PIL import Image
from collections import deque
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from utils.helpers import mat2euler


def index_sampler(traj_dir, sequence_len):
    traj_len = len(fnmatch.filter(os.listdir(traj_dir), '*.*'))
    if traj_len <= sequence_len:
        raise ValueError("trajectory length must be greater than the sequence length")
    last_valid_index = traj_len - sequence_len - 1
    index = random.randint(0, last_valid_index)
    return index


class BaseReplayBuffer(Dataset):
    """Base Replay Buffer class for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
        data_dir: directory containing the data
        sequence_len: sequence length for sampling
    """

    def __init__(self, capacity, data_dir, sequence_len):
        self.data_dir = data_dir
        self.sequence_len = sequence_len
        self.rgb_path = os.path.join(self.data_dir, "rgb")
        self.buffer = deque(maxlen=capacity)
        self.convert_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return os.path.join(self.data_dir, f"traj{idx}")

    def append(self, experience):
        """Add experience to the buffer.

        Args:
            experience: trajectory path
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        rgb_batch = []
        depth_batch = []
        joint_batch = []
        action_batch = []
        for idx in indices:
            rgb_in, depth_in, joint_in, act_in = self.experience_collector(self.buffer[idx])
            rgb_batch.append(rgb_in)
            depth_batch.append(depth_in)
            joint_batch.append(joint_in)
            action_batch.append(act_in)
        return torch.stack(rgb_batch), torch.stack(depth_batch), torch.stack(joint_batch), torch.stack(action_batch)


class ReplayBufferEuler(BaseReplayBuffer):
    def data_collector(self, rgb_dir, step_index):
        # image (rgb and depth)
        convert_tensor = transforms.ToTensor()
        rgb_img = Image.open(rgb_dir + "/{}.png".format(step_index))
        rgb_img = convert_tensor(rgb_img)
        traj_dir = rgb_dir[:-4]
        depth_dir = os.path.join(traj_dir, "depth")
        depth_img = Image.open(depth_dir + "/{}.png".format(step_index))
        depth_img = convert_tensor(depth_img)  
        #joint
        pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)
        joint_info = pose_file.item()[step_index]["joint_pos"]
        #action
        current_ee_pose_SM = pose_file.item()[step_index]["ee_pose"]
        t_ee_pose = current_ee_pose_SM.t.reshape(1, 3)
        R_ee_pose = current_ee_pose_SM.R
        R_ee_pose = mat2euler(R_ee_pose).reshape(1, 3)
        current_ee_pose = np.concatenate((t_ee_pose, R_ee_pose), axis=-1)
        return rgb_img, depth_img, joint_info, current_ee_pose
    
    def experience_collector(self, traj_dir):
        rgb_dir = os.path.join(traj_dir, "rgb")
        step_index = index_sampler(rgb_dir, self.sequence_len + 1)
        rgb_data, depth_data, joint_data, ee_pose_data, gripper_pos, action = [], [], [], [], [], []
        for i in range(self.sequence_len + 1):
            rgb_step, depth_step, joint_step, ee_pose_step = self.data_collector(rgb_dir, step_index + i)
            rgb_data.append(rgb_step)
            depth_data.append(depth_step)
            joint_data.append(torch.tensor(joint_step))
            ee_pose_data.append(ee_pose_step)
            gripper_pos.append(joint_step[-2] + joint_step[-1])
        rgb_data.pop()
        depth_data.pop()
        joint_data.pop()
        gripper_pos = gripper_pos[1:]
        ee_pose_data = np.array(ee_pose_data)
        old_action = ee_pose_data[1:] - ee_pose_data[:-1]
        for j in range(len(gripper_pos)):
            action_to_append = np.concatenate((old_action[j], np.array([0.8 if gripper_pos[j] > 0.06 else 0.0]).reshape(1,1)), axis=1)
            action.append(action_to_append)
        return torch.stack(rgb_data), torch.stack(depth_data), torch.stack(joint_data), torch.tensor(action)


class ReplayBufferAbsRot(BaseReplayBuffer):
    def data_collector(self, rgb_dir, step_index):
        # image (rgb and depth)
        convert_tensor = transforms.ToTensor()
        rgb_img = Image.open(rgb_dir + "/{}.png".format(step_index))
        rgb_img = convert_tensor(rgb_img)
        traj_dir = rgb_dir[:-4]
        depth_dir = os.path.join(traj_dir, "depth")
        depth_img = Image.open(depth_dir + "/{}.png".format(step_index))
        depth_img = convert_tensor(depth_img) 
        #joint
        pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)
        joint_info = pose_file.item()[step_index]["joint_pos"]
        #action
        current_ee_pose = pose_file.item()[step_index]["ee_pose"]
        return rgb_img, depth_img, joint_info, current_ee_pose

    def experience_collector(self, traj_dir):
        rgb_dir = os.path.join(traj_dir, "rgb")
        step_index = index_sampler(rgb_dir, self.sequence_len + 1)
        rgb_data, depth_data, joint_data, action_data, ext_action_data, gripper_pos = [], [], [], [], [], []
        for i in range(self.sequence_len + 1):
            rgb_step, depth_step, joint_step, ee_pose_step = self.data_collector(rgb_dir, step_index + i)
            rgb_data.append(rgb_step)
            depth_data.append(depth_step)
            joint_data.append(torch.tensor(joint_step))
            action_data.append(np.concatenate((ee_pose_step.t.reshape(1,3), ee_pose_step.R.reshape(1,9)), axis=-1))
            gripper_pos.append(joint_step[-2] + joint_step[-1])
        rgb_data.pop()
        depth_data.pop()
        joint_data.pop()
        gripper_pos = gripper_pos[1:]
        action_data = np.array(action_data)[1:]
        for j, action in enumerate(action_data):
            ext_action_to_append = np.concatenate((action, np.array([0.8 if gripper_pos[j] > 0.06 else 0.0]).reshape(1,1)), axis=1)
            ext_action_data.append(ext_action_to_append)
        return torch.stack(rgb_data), torch.stack(depth_data), torch.stack(joint_data), torch.tensor(ext_action_data)


class ReplayBufferTwist(BaseReplayBuffer):
    def data_collector(self, rgb_dir, step_index):
        ## image (rgb and depth)
        convert_tensor = transforms.ToTensor()
        rgb_img = Image.open(rgb_dir + "/{}.png".format(step_index))
        rgb_img = convert_tensor(rgb_img)
        traj_dir = rgb_dir[:-4]
        depth_dir = os.path.join(traj_dir, "depth")
        depth_img = Image.open(depth_dir + "/{}.png".format(step_index))
        depth_img = convert_tensor(depth_img)              
        ##joint
        pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)
        joint_info = pose_file.item()[step_index]["joint_pos"]
        ##action
        current_twist = pose_file.item()[step_index]["ee_twist"]
        return rgb_img, depth_img, joint_info, current_twist

    def experience_collector(self, traj_dir):
        rgb_dir = os.path.join(traj_dir, "rgb")
        step_index = index_sampler(rgb_dir, self.sequence_len)
        rgb_data, depth_data, joint_data, ee_twist_data, gripper_pos, action_data = [], [], [], [], [], []
        for i in range(self.sequence_len):
            rgb_step, depth_step, joint_step, ee_twist_step = self.data_collector(rgb_dir, step_index + i)
            rgb_data.append(rgb_step)
            depth_data.append(depth_step)
            joint_data.append(torch.tensor(joint_step))
            ee_twist_data.append(ee_twist_step)
            gripper_pos.append(joint_step[-2] + joint_step[-1])
        for j, action in enumerate(ee_twist_data):
            action_to_append = np.concatenate((action.reshape(1, 6), np.array([0.8 if gripper_pos[j] > 0.06 else 0.0]).reshape(1,1)), axis=1)
            action_data.append(action_to_append)
        return torch.stack(rgb_data), torch.stack(depth_data), torch.stack(joint_data), torch.tensor(action_data)


class ReplayBufferTwistFeedback(BaseReplayBuffer):
    def data_collector(self, rgb_dir, step_index):
        ## image (rgb and depth)
        convert_tensor = transforms.ToTensor()
        rgb_img = Image.open(rgb_dir + "/{}.png".format(step_index))
        rgb_img = convert_tensor(rgb_img)
        traj_dir = rgb_dir[:-4]
        depth_dir = os.path.join(traj_dir, "depth")
        depth_img = Image.open(depth_dir + "/{}.png".format(step_index))
        depth_img = convert_tensor(depth_img)              
        ##joint
        pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)
        joint_info = pose_file.item()[step_index]["joint_pos"]
        ##action
        feedback_file = np.load(traj_dir + "/feedback.npy", allow_pickle=True)
        current_twist = feedback_file.item()[step_index]
        return rgb_img, depth_img, joint_info, current_twist

    def experience_collector(self, traj_dir):
        rgb_dir = os.path.join(traj_dir, "rgb")
        step_index = index_sampler(rgb_dir, self.sequence_len)
        rgb_data, depth_data, joint_data, action_data = [], [], [], []
        for i in range(self.sequence_len):
            rgb_step, depth_step, joint_step, ee_twist_step = self.data_collector(rgb_dir, step_index + i)
            rgb_data.append(rgb_step)
            depth_data.append(depth_step)
            joint_data.append(torch.tensor(joint_step))
            action_data.append(ee_twist_step)
        return torch.stack(rgb_data), torch.stack(depth_data), torch.stack(joint_data), torch.tensor(action_data)


class BPTTReplayBufferTwist(BaseReplayBuffer):
    def __init__(self, capacity, data_dir):
        super().__init__(capacity, data_dir, sequence_len=0)

    def data_collector(self, rgb_dir, step_index):
        ## image (rgb and depth)
        convert_tensor = transforms.ToTensor()
        rgb_img = Image.open(rgb_dir + "/{}.png".format(step_index))
        rgb_img = convert_tensor(rgb_img)
        traj_dir = rgb_dir[:-4]
        depth_dir = os.path.join(traj_dir, "depth")
        depth_img = Image.open(depth_dir + "/{}.png".format(step_index))
        depth_img = convert_tensor(depth_img)              
        ##joint
        pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)
        joint_info = pose_file.item()[step_index]["joint_pos"]
        ##action
        current_twist = pose_file.item()[step_index]["ee_twist"]
        return rgb_img, depth_img, joint_info, current_twist

    def experience_collector(self, traj_dir):
        rgb_dir = os.path.join(traj_dir, "rgb")
        rgb_data, depth_data, joint_data, ee_twist_data, gripper_pos, action_data = [], [], [], [], [], []
        for i in range(len(os.listdir(rgb_dir))):
            rgb_step, depth_step, joint_step, ee_twist_step = self.data_collector(rgb_dir, i)
            rgb_data.append(rgb_step)
            depth_data.append(depth_step)
            joint_data.append(torch.tensor(joint_step))
            ee_twist_data.append(ee_twist_step)
            gripper_pos.append(joint_step[-2] + joint_step[-1])
        for j, action in enumerate(ee_twist_data):
            action_to_append = np.concatenate((action.reshape(1, 6), np.array([1.0 if gripper_pos[j] > 0.06 else 0.0]).reshape(1,1)), axis=1)
            action_data.append(action_to_append)
        return torch.stack(rgb_data), torch.stack(depth_data), torch.stack(joint_data), torch.tensor(action_data)

    def padded_sample(self, batch_size, max_length):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        rgb_batch, depth_batch, joint_batch, action_batch = [], [], [], []
        for idx in indices:
            rgb_in, depth_in, joint_in, act_in = self.experience_collector(self.buffer[idx])
            rgb_batch.append(rgb_in)
            depth_batch.append(depth_in)
            joint_batch.append(joint_in)
            action_batch.append(act_in)
        # Pad the sequences to the maximum length
        def pad_sequence(seq, max_length):
            return np.pad(seq, [(0, max_length - seq.shape[0])] + [(0, 0)] * (len(seq.shape) - 1), mode='constant')
        padded_rgb_batch = [pad_sequence(seq, max_length) for seq in rgb_batch]
        padded_depth_batch = [pad_sequence(seq, max_length) for seq in depth_batch]
        padded_joint_batch = [pad_sequence(seq, max_length) for seq in joint_batch]
        padded_action_batch = [pad_sequence(seq, max_length) for seq in action_batch]
        return torch.tensor(np.array(padded_rgb_batch)), torch.tensor(np.array(padded_depth_batch)), \
               torch.tensor(np.array(padded_joint_batch)), torch.tensor(np.array(padded_action_batch))