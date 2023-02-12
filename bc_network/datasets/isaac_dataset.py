import os
import fnmatch
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class TrajectoriesDataset(Dataset):
    def __init__(self, data_dir, sequence_len):
        self.data_dir = data_dir
        self.sequence_len = sequence_len
        self.rgb_path = os.path.join(self.data_dir, "/rgb")
        self.depth_path = os.path.join(self.data_dir, "/depth")
        self.initial_index = self.initial_index_finder(self.data_dir)
        return

    def __len__(self):
        return len(next(os.walk(self.data_dir))[1])

    def __getitem__(self, idx):
        rgb_dir = os.path.join(self.data_dir, "traj{}/rgb".format(self.initial_index + idx))
        image_data = []
        joint_data = []
        ee_pose_data = []
        step_index = index_sampler(rgb_dir, self.sequence_len)
        for i in range(self.sequence_len + 1):
            image_step, joint_step, ee_pose_step = self.data_collector(rgb_dir, step_index + i)
            image_data.append(image_step)
            joint_data.append(torch.tensor(joint_step))
            ee_pose_data.append(ee_pose_step) 
 
        image_data.pop()
        joint_data.pop()
        ee_pose_data = np.array(ee_pose_data)
        action = ee_pose_data[:-1] - ee_pose_data[1:]
        return image_data, joint_data, action

    def data_collector(self, rgb_dir, step_index):
        convert_tensor = transforms.ToTensor()
        rgb_img = Image.open(rgb_dir + "/{}.jpeg".format(step_index))
        rgb_img = convert_tensor(rgb_img)
        traj_dir = rgb_dir[:-4]
        depth_dir = os.path.join(traj_dir, "depth")
        depth_img = Image.open(depth_dir + "/{}.jpeg".format(step_index))
        depth_img = convert_tensor(depth_img)
        im_info = torch.cat((rgb_img, depth_img), dim=0)
        pose_file = np.load(traj_dir + "/pose.npy", allow_pickle=True)
        joint_info = pose_file.item()[step_index]["joint_pos"]
        current_ee_pose_SM = pose_file.item()[step_index]["ee_pose"]
        t_ee_pose = current_ee_pose_SM.t.reshape(1, 3)
        R_ee_pose = current_ee_pose_SM.R
        R_ee_pose = rotation_angles(R_ee_pose).reshape(1, 3)
        current_ee_pose = np.concatenate((t_ee_pose, R_ee_pose, np.array([0]).reshape(1,1)), axis=-1)
        return im_info, joint_info, current_ee_pose

    def initial_index_finder(self, data_dir):
        folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        folders.sort()
        first_folder = folders[0] if folders else None
        index = int(first_folder[4:])
        return index
    
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

