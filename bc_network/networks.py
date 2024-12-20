import torch
import roma
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class BasePolicy(nn.Module):
    def __init__(self, config, device):
        super(BasePolicy, self).__init__()
        self._device = device
        self.lstm_dim1 = config["resnet_visual_embedding_dim"] + config["proprio_dim"]
        self.lstm_dim2 = self.lstm_dim1
        # Load pre-trained ResNet18 models
        self.resnet_rgb = resnet18(pretrained=True).to(device)
        self.resnet_rgb = nn.Sequential(*list(self.resnet_rgb.children())[:-2])
        self.resnet_depth = resnet18(pretrained=True).to(device)
        self.resnet_depth.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
        self.resnet_depth = nn.Sequential(*list(self.resnet_depth.children())[:-2])
        # Conv layers
        self.conv_rgb1 = nn.Conv2d(512, 3, kernel_size=3, padding=1, stride=1).to(device)
        self.conv_depth1 = nn.Conv2d(512, 3, kernel_size=1, padding=1, stride=1).to(device)
        # LSTM layers
        self.lstm1 = nn.LSTM(self.lstm_dim1, self.lstm_dim1).to(device)
        self.linear_out = nn.Linear(self.lstm_dim1, config["action_dim"]).to(device)
        self.lstm2 = nn.LSTM(self.lstm_dim1, self.lstm_dim2).to(device)
        self.linear_out2 = nn.Linear(self.lstm_dim2, config["action_dim"]).to(device)
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        self.loss = nn.MSELoss().to(device)

    @property
    def device(self):
        return self._device

    def forward_step(self, rgb_obs, depth_obs, proprio_obs, lstm_state1, lstm_state2):
        vis_encoding_rgb = self.resnet_rgb(rgb_obs)
        vis_encoding_rgb = F.elu(self.conv_rgb1(vis_encoding_rgb))
        vis_encoding_rgb = torch.flatten(vis_encoding_rgb, start_dim=1)
        vis_encoding_depth = self.resnet_depth(depth_obs)
        vis_encoding_depth = F.elu(self.conv_depth1(vis_encoding_depth))
        vis_encoding_depth = torch.flatten(vis_encoding_depth, start_dim=1)
        low_dim_input = torch.cat((vis_encoding_rgb, vis_encoding_depth, proprio_obs), dim=-1).unsqueeze(0)
        lstm_out1, (h1, c1) = self.lstm1(low_dim_input, lstm_state1)
        lstm_out2, (h2, c2) = self.lstm2(lstm_out1, lstm_state2)
        lstm_state1, lstm_state2 = (h1, c1), (h2, c2)
        out = torch.tanh(self.linear_out(lstm_out2))
        return out, lstm_state1, lstm_state2

    def forward(self, rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj):
        losses_t, losses_r, losses_g = [], [], []
        lstm_state1, lstm_state2 = None, None
        for idx in range(len(proprio_obs_traj)):
            output, lstm_state1, lstm_state2 = self.forward_step(rgb_obs_traj[idx], depth_obs_traj[idx], proprio_obs_traj[idx], lstm_state1, lstm_state2)
            losses_t.append(self.loss(output[:,:,:3], action_traj[idx][:,:,:3].permute(1, 0, 2)))
            losses_r.append(self.loss(output[:,:,3:-1], action_traj[idx][:,:,3:-1].permute(1, 0, 2)))
            losses_g.append(self.loss(output[:,:,-1], action_traj[idx][:,:,-1].permute(1, 0)))
        total_loss_t = sum(losses_t) / len(losses_t)
        total_loss_r = sum(losses_r) / len(losses_r)
        total_loss_g = sum(losses_g) / len(losses_g)
        return total_loss_t, total_loss_r, total_loss_g

    def update_params(self, rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj):
        rgb_obs, depth_obs, proprio_obs, action = map(lambda x: x.to(self.device).float(), [rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj])
        self.optimizer.zero_grad()
        loss_t, loss_r, loss_g = self.forward(rgb_obs, depth_obs, proprio_obs, action)
        total_loss = loss_t + loss_r + loss_g
        total_loss.backward()
        self.optimizer.step()
        return {"total loss": total_loss, "translation loss": loss_t, "rotation loss": loss_r, "gripper loss": loss_g}

    def predict(self, rgb_obs, depth_obs, proprio_obs, lstm_state1, lstm_state2):
        rgb_obs_th = torch.tensor(rgb_obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        depth_obs_th = torch.tensor(depth_obs, dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_th, lstm_state1, lstm_state2 = self.forward_step(rgb_obs_th, depth_obs_th, proprio_obs_th, lstm_state1, lstm_state2)
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state1, lstm_state2


class PolicyEuler(BasePolicy):
    def __init__(self, config, device):
        super(PolicyEuler, self).__init__(config, device)

    def update_params(self, rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj):
        rgb_obs, depth_obs, proprio_obs, action = map(lambda x: x.to(self.device).float(), [rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj])
        self.optimizer.zero_grad()
        loss_t, loss_r, loss_g = self.forward(rgb_obs, depth_obs, proprio_obs, action)
        total_loss = (loss_t * 1000000) + loss_r + (loss_g / 100)
        total_loss.backward()
        self.optimizer.step()
        return {"total loss": total_loss, "translation loss": loss_t, "rotation loss": loss_r, "gripper loss": loss_g}


class PolicyAbsRot(BasePolicy):
    def __init__(self, config, device):
        super(PolicyAbsRot, self).__init__(config, device)

    def forward_step(self, rgb_obs, depth_obs, proprio_obs, lstm_state1, lstm_state2):
        out, lstm_state1, lstm_state2 = super().forward_step(rgb_obs, depth_obs, proprio_obs, lstm_state1, lstm_state2)
        modified_out = torch.zeros(out.shape[0], out.shape[1], out.shape[2]).to(self.device)
        modified_out[:, :, 0] = out[:, :, 0] * 2
        modified_out[:, :, 1:] = out[:, :, 1:]
        return modified_out, lstm_state1, lstm_state2

    def forward(self, rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj):
        losses_t, losses_r, losses_g = [], [], []
        lstm_state1, lstm_state2 = None, None
        for idx in range(len(proprio_obs_traj)):
            output, lstm_state1, lstm_state2 = self.forward_step(rgb_obs_traj[idx], depth_obs_traj[idx], proprio_obs_traj[idx], lstm_state1, lstm_state2)
            rot_lin = output[:, :, 3:-1]
            rot_mat = rot_lin.view(output.shape[0], output.shape[1], 3, 3)
            rot_mat = roma.special_procrustes(rot_mat)
            rot_lin = rot_mat.view(output.shape[0], output.shape[1], 9)
            output_new = output.clone()
            output_new[:, :, 3:-1] = rot_lin
            losses_t.append(self.loss(output_new[:, :, :3], action_traj[idx][:, :, :3].permute(1, 0, 2)))
            losses_r.append(self.loss(output_new[:, :, 3:-1], action_traj[idx][:, :, 3:-1].permute(1, 0, 2)))
            losses_g.append(self.loss(output_new[:, :, -1], action_traj[idx][:, :, -1].permute(1, 0)))
        total_loss_t = sum(losses_t) / len(losses_t)
        total_loss_r = sum(losses_r) / len(losses_r)
        total_loss_g = sum(losses_g) / len(losses_g)
        return total_loss_t.float(), total_loss_r.float(), total_loss_g.float()

    def update_params(self, rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj):
        rgb_obs, depth_obs, proprio_obs, action = map(lambda x: x.to(self.device).float(), [rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj])
        self.optimizer.zero_grad()
        loss_t, loss_r, loss_g = self.forward(rgb_obs, depth_obs, proprio_obs, action)
        total_loss = loss_t + (15 * loss_r) + loss_g
        total_loss.backward()
        self.optimizer.step()
        return {"total loss": total_loss, "translation loss": loss_t, "rotation loss": loss_r, "gripper loss": loss_g}


class PolicyTwist(BasePolicy):
    def __init__(self, config, device):
        super(PolicyTwist, self).__init__(config, device)

    def forward_step(self, rgb_obs, depth_obs, proprio_obs, lstm_state1, lstm_state2):
        out, lstm_state1, lstm_state2 = super().forward_step(rgb_obs, depth_obs, proprio_obs, lstm_state1, lstm_state2)
        modified_out = torch.zeros(out.shape[0], out.shape[1], out.shape[2]).to(self.device)
        modified_out[:, :, :3] = out[:, :, :3] * 0.1
        modified_out[:, :, 3:-1] = out[:, :, 3:-1] * 0.5
        modified_out[:, :, -1] = out[:, :, -1]
        return modified_out, lstm_state1, lstm_state2


class PolicyBPTT(PolicyTwist):
    def __init__(self, config, device):
        self.bptt_chunk = config["bptt_chunk"]
        super(PolicyBPTT, self).__init__(config, device)

    def forward(self, rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj, lstm_state1, lstm_state2):
        losses_t, losses_r, losses_g = [], [], []
        for rgb_obs, depth_obs, proprio_obs, action in zip(rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj):
            output, lstm_state1, lstm_state2 = self.forward_step(rgb_obs, depth_obs, proprio_obs, lstm_state1, lstm_state2)
            losses_t.append(self.loss(output[:,:,:3], action[:,:,:3].permute(1, 0, 2)))
            losses_r.append(self.loss(output[:,:,3:-1], action[:,:,3:-1].permute(1, 0, 2)))
            losses_g.append(self.loss(output[:,:,-1], action[:,:,-1].permute(1, 0)))
        total_loss_t = sum(losses_t) / len(losses_t)
        total_loss_r = sum(losses_r) / len(losses_r)
        total_loss_g = sum(losses_g) / len(losses_g)
        return total_loss_t.float(), total_loss_r.float(), total_loss_g.float()

    def update_params(self, rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj):
        rgb_obs, depth_obs, proprio_obs, action = map(lambda x: x.to(self.device).float(), [rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj])
        lstm_state1 = None
        lstm_state2 = None
        total_loss, loss_t, loss_r, loss_g = 0, 0, 0, 0
        num_chunks = int(np.ceil(rgb_obs.shape[1] / self.bptt_chunk))
        iterator = zip(rgb_obs.chunk(num_chunks, dim=1), depth_obs.chunk(num_chunks, dim=1),
                       proprio_obs.chunk(num_chunks, dim=1), action.chunk(num_chunks, dim=1))
        for rgb_obs_chunk, depth_obs_chunk, proprio_obs_chunk, action_chunk in iterator:
            self.optimizer.zero_grad()
            chunk_loss_t, chunk_loss_r, chunk_loss_g = self.forward(rgb_obs_chunk, depth_obs_chunk, proprio_obs_chunk, action_chunk, lstm_state1, lstm_state2)
            chunk_total_loss = chunk_loss_t + chunk_loss_r + chunk_loss_g
            chunk_total_loss.backward()
            self.optimizer.step()
            loss_t += chunk_loss_t.item()
            loss_r += chunk_loss_r.item()
            loss_g += chunk_loss_g.item()
            total_loss += chunk_total_loss.item()
        loss_t /= len(rgb_obs)
        loss_r /= len(rgb_obs)
        loss_g /= len(rgb_obs)
        total_loss /= len(rgb_obs)
        return {"total loss": total_loss, "translation loss": loss_t, "rotation loss": loss_r, "gripper loss": loss_g}


def binary_gripper(gripper_action):
    return 0.8 if gripper_action >= 0.4 else 0.0