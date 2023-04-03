import os
import fnmatch
import random
import numpy as np
from PIL import Image
from collections import deque
import torch
from torchvision import transforms
from torch.utils.data import Dataset
    
class ReplayBuffer(Dataset):
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity, data_dir, sequence_len):
        self.data_dir = data_dir
        self.sequence_len = sequence_len
        self.rgb_path = os.path.join(self.data_dir, "/rgb")
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return os.path.join(self.data_dir, "traj{}".format(idx))


    def append(self, experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: trajectory path
        """
        self.buffer.append(experience)

    def experience_collector(self, traj_dir):
        rgb_dir = os.path.join(traj_dir, "rgb")
        step_index = index_sampler(rgb_dir, self.sequence_len + 1)
        image_data = []
        joint_data = []
        ee_pose_data = []
        gripper_pos = []
        action = []
        for i in range(self.sequence_len + 1):
            image_step, joint_step, ee_pose_step = self.data_collector(rgb_dir, step_index + i)
            image_data.append(image_step)
            joint_data.append(torch.tensor(joint_step))
            ee_pose_data.append(ee_pose_step)
            gripper_pos.append(joint_step[-2] + joint_step[-1])
        image_data.pop()
        joint_data.pop()
        gripper_pos = gripper_pos[1:]
        ee_pose_data = np.array(ee_pose_data)

        old_action = ee_pose_data[1:] - ee_pose_data[:-1]
        for j in range(len(gripper_pos)):
            if gripper_pos[j] > 0.04:
                action.append(np.concatenate((old_action[j], np.array([0.8]).reshape(1,1)), axis=1))
            else:
                action.append(np.concatenate((old_action[j], np.array([-0.8]).reshape(1,1)), axis=1))
        
        # for i in range(ee_pose_data.shape[0] - 1):
        #     diff_pos = ee_pose_data[i + 1,:,:3] - ee_pose_data[i,:,:3]
        #     # Computing the relative quaternion
        #     diff_rel = Rotation.from_quat(ee_pose_data[i,:,3:]).inv() * Rotation.from_quat(ee_pose_data[i+1,:,3:])
        #     old_action = np.concatenate((diff_pos, diff_rel.as_quat()), axis=1)
        #     if gripper_pos[i] > 0.04:
        #         action.append(np.concatenate((old_action, np.array([0.8]).reshape(1,1)), axis=1))
        #     else:
        #         action.append(np.concatenate((old_action, np.array([-0.8]).reshape(1,1)), axis=1))
        
        return torch.stack(image_data), torch.stack(joint_data), torch.tensor(action)

    def data_collector(self, rgb_dir, step_index):
        # image (rgb and depth)
        convert_tensor = transforms.ToTensor()
        rgb_img = Image.open(rgb_dir + "/{}.png".format(step_index))
        rgb_img = convert_tensor(rgb_img)
        traj_dir = rgb_dir[:-4]
        depth_dir = os.path.join(traj_dir, "depth")
        depth_img = Image.open(depth_dir + "/{}.png".format(step_index))
        depth_img = convert_tensor(depth_img)              
        im_info = torch.cat((rgb_img, depth_img), dim=0)
        #joint
        pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)
        joint_info = pose_file.item()[step_index]["joint_pos"]
        #action
        current_ee_pose_SM = pose_file.item()[step_index]["ee_pose"]
        t_ee_pose = current_ee_pose_SM.t.reshape(1, 3)
        R_ee_pose = current_ee_pose_SM.R
        # # for Euler start removing here
        # r_ee_matrix = Rotation.from_matrix(R_ee_pose)
        # r_ee_quat = r_ee_matrix.as_quat()
        # # finish removing here and uncomment the below line and change the second element of current_ee_pose
        R_ee_pose = rotation_angles(R_ee_pose).reshape(1, 3)
        current_ee_pose = np.concatenate((t_ee_pose, R_ee_pose), axis=-1)

        return im_info, joint_info, current_ee_pose

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        image_batch = []
        joint_batch = []
        action_batch = []
        for idx in indices:
            img_in, joint_in, act_in = self.experience_collector(self.buffer[idx])
            image_batch.append(img_in)
            joint_batch.append(joint_in)
            action_batch.append(act_in)
        return torch.stack(image_batch), torch.stack(joint_batch), torch.stack(action_batch)
    


def index_sampler(traj_dir, sequence_len):
    traj_len = len(fnmatch.filter(os.listdir(traj_dir), '*.*'))
    if traj_len <= sequence_len:
        raise "trajectory length must be greater than the sequence length" 
    last_valid_index = int(traj_len - sequence_len - 1)
    index = random.randint(0, last_valid_index)
    return index


def rotation_angles(matrix):
    """
    input
        matrix = 3x3 rotation matrix (numpy array)
        oreder(str) = rotation order of x, y, z : e.g, rotation XZY -- 'xzy'
    output
        theta1, theta2, theta3 = rotation angles in rotation order
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    theta1 = np.arctan(-r23 / r33)
    theta2 = np.arctan(r13 * np.cos(theta1) / r33)
    theta3 = np.arctan(-r12 / r11)

    return np.array((theta1, theta2, theta3))



def quaternion_difference(q1, q2):
    """
    Calculate the difference in orientation between two quaternions
    """
    # Normalize the quaternions to ensure they have unit magnitude
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Calculate the dot product between the two quaternions
    dot_product = np.dot(q1, q2)
    
    # If the dot product is negative, the quaternions are on opposite sides of the rotation
    # sphere and the difference is greater than pi radians
    if dot_product < 0:
        q1 = -q1
        dot_product = -dot_product
    
    # Calculate the angle between the quaternions
    angle = 2 * np.arccos(dot_product)
    
    # Calculate the axis of rotation between the quaternions
    axis = np.cross(q1, q2)
    axis = axis / np.linalg.norm(axis)
    
    # Calculate the quaternion that represents the difference in orientation
    diff_quat = np.array([np.cos(angle/2), axis[0]*np.sin(angle/2), axis[1]*np.sin(angle/2), axis[2]*np.sin(angle/2)])
    
    return diff_quat


class ReplayBuffer_abs_rot(Dataset):
    """Replay Buffer for storing past experiences allowing the agent to learn from them. 
    The output is exact not difference and angles are presented as rotation matrix not Euler. 

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity, data_dir, sequence_len):
        self.data_dir = data_dir
        self.sequence_len = sequence_len
        self.rgb_path = os.path.join(self.data_dir, "/rgb")
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return os.path.join(self.data_dir, "traj{}".format(idx))


    def append(self, experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: trajectory path
        """
        self.buffer.append(experience)

    def experience_collector(self, traj_dir):
        rgb_dir = os.path.join(traj_dir, "rgb")
        step_index = index_sampler(rgb_dir, self.sequence_len + 1)
        image_data = []
        joint_data = []
        action_data = []
        ext_action_data = []
        gripper_pos = []
        for i in range(self.sequence_len + 1):
            image_step, joint_step, ee_pose_step = self.data_collector(rgb_dir, step_index + i)
            image_data.append(image_step)
            joint_data.append(torch.tensor(joint_step))
            action_data.append(np.concatenate((ee_pose_step.t.reshape(1,3), ee_pose_step.R.reshape(1,9)), axis=-1))
            gripper_pos.append(joint_step[-2] + joint_step[-1])
        image_data.pop()
        joint_data.pop()
        gripper_pos = gripper_pos[1:]
        action_data = np.array(action_data)[1:]

        for j, action in enumerate(action_data):
            if gripper_pos[j] > 0.04:
                ext_action_data.append(np.concatenate((action, np.array([0.8]).reshape(1,1)), axis=1))
            else:
                ext_action_data.append(np.concatenate((action, np.array([-0.8]).reshape(1,1)), axis=1))
        ext_action_data = np.array(ext_action_data)
        return torch.stack(image_data), torch.stack(joint_data), torch.tensor(ext_action_data)

    def data_collector(self, rgb_dir, step_index):
        # image (rgb and depth)
        convert_tensor = transforms.ToTensor()
        rgb_img = Image.open(rgb_dir + "/{}.png".format(step_index))
        rgb_img = convert_tensor(rgb_img)
        traj_dir = rgb_dir[:-4]
        depth_dir = os.path.join(traj_dir, "depth")
        depth_img = Image.open(depth_dir + "/{}.png".format(step_index))
        depth_img = convert_tensor(depth_img)              
        im_info = torch.cat((rgb_img, depth_img), dim=0)
        #joint
        pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)
        joint_info = pose_file.item()[step_index]["joint_pos"]
        #action
        current_ee_pose = pose_file.item()[step_index]["ee_pose"]

        return im_info, joint_info, current_ee_pose

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        image_batch = []
        joint_batch = []
        action_batch = []
        for idx in indices:
            img_in, joint_in, pose_in = self.experience_collector(self.buffer[idx])
            image_batch.append(img_in)
            joint_batch.append(joint_in)
            action_batch.append(pose_in)
        return torch.stack(image_batch), torch.stack(joint_batch), torch.stack(action_batch)
