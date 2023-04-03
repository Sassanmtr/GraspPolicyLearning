import torch
import roma
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal



class Policy(nn.Module):
    def __init__(self, config, device):
        super(Policy, self).__init__()
        self._device = device
        lstm_dim = config["visual_embedding_dim"] + config["proprio_dim"]
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=2, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.conv3 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.lstm = nn.LSTM(lstm_dim, lstm_dim).to(device)
        self.linear_out = nn.Linear(lstm_dim, config["action_dim"]).to(device)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        # self.std = 0.1 * torch.ones(config["action_dim"], dtype=torch.float32)
        self.std1 = 0.1 * torch.ones(3, dtype=torch.float32)
        self.std2 = 0.1 * torch.ones(3, dtype=torch.float32) # change it to 3 for Euler angles (4 for quaternion)
        self.std1 = self.std1.to(self.device)
        self.std2 = self.std2.to(self.device)

        return

    @property
    def device(self):
        return self._device

    def forward_step(self, camera_obs, proprio_obs, lstm_state):
        vis_encoding = F.elu(self.conv1(camera_obs))
        vis_encoding = F.elu(self.conv2(vis_encoding))
        vis_encoding = F.elu(self.conv3(vis_encoding))
        vis_encoding = torch.flatten(vis_encoding, start_dim=1)
        low_dim_input = torch.cat((vis_encoding, proprio_obs), dim=-1).unsqueeze(0)
        lstm_out, (h, c) = self.lstm(low_dim_input, lstm_state)
        lstm_state = (h, c)
        out = torch.tanh(self.linear_out(lstm_out))
        return out, lstm_state

    def forward(self, camera_obs_traj, proprio_obs_traj, action_traj):
        losses_t = []
        losses_r = []
        losses_g = []
        lstm_state = None
        for idx in range(len(proprio_obs_traj)):
            mu, lstm_state = self.forward_step(
                camera_obs_traj[idx], proprio_obs_traj[idx], lstm_state
            )
            distribution_t = Normal(mu[:,:,:3], self.std1)
            distribution_r = Normal(mu[:,:,3:-1], self.std2)
            distribution_g = Normal(mu[:,:,-1], self.std1[-1])
            log_prob_t = distribution_t.log_prob(action_traj[idx][:,:,:3].permute(1, 0, 2))
            log_prob_r = distribution_r.log_prob(action_traj[idx][:,:,3:-1].permute(1, 0, 2))
            log_prob_g = distribution_g.log_prob(action_traj[idx][:,:,-1].permute(1, 0))
            loss_t = -log_prob_t
            loss_r = -log_prob_r
            loss_g = -log_prob_g
            losses_t.append(loss_t)
            losses_r.append(loss_r)
            losses_g.append(loss_g)
        total_loss_t = torch.cat(losses_t).mean()
        total_loss_r = torch.cat(losses_r).mean()
        total_loss_g = torch.cat(losses_g).mean()
        return total_loss_t, total_loss_r, total_loss_g

    def update_params(
        self, camera_obs_traj, proprio_obs_traj, action_traj
    ):
        camera_obs = camera_obs_traj.to(self.device)
        proprio_obs = proprio_obs_traj.to(self.device)
        action = action_traj.to(self.device)
        self.optimizer.zero_grad()
        loss_t, loss_r , loss_g= self.forward(camera_obs, proprio_obs, action)
        total_loss = loss_t + loss_r #+ (loss_g/500)
        total_loss.backward()
        self.optimizer.step()
        training_metrics = {"total loss": total_loss, "translation loss": loss_t, "rotation loss": loss_r, "gripper loss": loss_g}
        return training_metrics

    def predict(self, camera_obs, proprio_obs, lstm_state):
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32).unsqueeze(0)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32).unsqueeze(0)
        camera_obs_th = camera_obs_th.to(self.device)
        proprio_obs_th = proprio_obs_th.to(self.device)
        with torch.no_grad():
            action_th, lstm_state = self.forward_step(
                camera_obs_th, proprio_obs_th, lstm_state
            )
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state




class Policy_abs_rot(nn.Module):
    def __init__(self, config, device):
        super(Policy_abs_rot, self).__init__()
        self._device = device
        lstm_dim = config["visual_embedding_dim"] + config["proprio_dim"]
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=2, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.conv3 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.lstm = nn.LSTM(lstm_dim, lstm_dim).to(device)
        self.linear_out = nn.Linear(lstm_dim, config["action_dim_abs_rot"]).to(device)
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.loss = nn.MSELoss().to(device)
        return

    @property
    def device(self):
        return self._device

    def forward_step(self, camera_obs, proprio_obs, lstm_state):
        vis_encoding = F.elu(self.conv1(camera_obs))
        vis_encoding = F.elu(self.conv2(vis_encoding))
        vis_encoding = F.elu(self.conv3(vis_encoding))
        vis_encoding = torch.flatten(vis_encoding, start_dim=1)
        low_dim_input = torch.cat((vis_encoding, proprio_obs), dim=-1).unsqueeze(0)
        lstm_out, (h, c) = self.lstm(low_dim_input, lstm_state)
        lstm_state = (h, c)
        out = torch.tanh(self.linear_out(lstm_out))
        return out, lstm_state


    def forward(self, camera_obs_traj, proprio_obs_traj, action_traj):
        losses_t = []
        losses_r = []
        losses_g = []
        lstm_state = None
        for idx in range(len(proprio_obs_traj)):
            output, lstm_state = self.forward_step(
                camera_obs_traj[idx], proprio_obs_traj[idx], lstm_state
            )
            rot_lin = output[:, :, 3:-1]
            # make it a 3 by 3 matrix
            rot_mat = rot_lin.view(output.shape[0], output.shape[1], 3, 3)
            # make it a rotation matrix
            rot_mat = roma.special_procrustes(rot_mat)
            # use the orthonormalized rotation matrix for loss computation
            rot_lin = rot_mat.view(output.shape[0], output.shape[1] , 9)
            output_new = output.clone()
            output_new[:, :, 3: -1] = rot_lin

            losses_t.append(self.loss(output_new[:,:,:3], action_traj[idx][:,:,:3].permute(1, 0, 2)))
            losses_r.append(self.loss(output_new[:,:,3:-1], action_traj[idx][:,:,3:-1].permute(1, 0, 2)))
            losses_g.append(self.loss(output_new[:,:,-1], action_traj[idx][:,:,-1].permute(1, 0)))
        total_loss_t = sum(losses_t) #/ len(losses_t)
        total_loss_r = sum(losses_r) #/ len(losses_r)
        total_loss_g = sum(losses_g) #/ len(losses_g)
        return total_loss_t.float(), total_loss_r.float(), total_loss_g.float()
    

    def update_params(
        self, camera_obs_traj, proprio_obs_traj, action_traj
    ):
        camera_obs = camera_obs_traj.to(self.device).float()
        proprio_obs = proprio_obs_traj.to(self.device).float()
        action = action_traj.to(self.device).float()
        self.optimizer.zero_grad()
        loss_t, loss_r , loss_g= self.forward(camera_obs, proprio_obs, action)
        total_loss = loss_t + loss_r #+ (loss_g/500)
        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        training_metrics = {"total loss": total_loss, "translation loss": loss_t, "rotation loss": loss_r, "gripper loss": loss_g}
        return training_metrics

    def predict(self, camera_obs, proprio_obs, lstm_state):
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32).unsqueeze(0)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32).unsqueeze(0)
        camera_obs_th = camera_obs_th.to(self.device)
        proprio_obs_th = proprio_obs_th.to(self.device)
        with torch.no_grad():
            action_th, lstm_state = self.forward_step(
                camera_obs_th, proprio_obs_th, lstm_state
            )
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state



def binary_gripper(gripper_action):
    if gripper_action >= 0.0:
        gripper_action = 0.8
    elif gripper_action < 0.0:
        gripper_action = -0.8
    return gripper_action
