import torch
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
        self.std = 0.1 * torch.ones(3, dtype=torch.float32)
        self.std = self.std.to(self.device)
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
        lstm_state = None
        for idx in range(len(proprio_obs_traj)):
            mu, lstm_state = self.forward_step(
                camera_obs_traj[idx], proprio_obs_traj[idx], lstm_state
            )
            distribution_t = Normal(mu[:,:,:3], self.std)
            distribution_r = Normal(mu[:,:,3:-1], self.std)
            log_prob_t = distribution_t.log_prob(action_traj[idx][:,:,:3].permute(1, 0, 2))
            log_prob_r = distribution_r.log_prob(action_traj[idx][:,:,3:-1].permute(1, 0, 2))
            loss_t = -log_prob_t
            loss_r = -log_prob_r
            losses_t.append(loss_t)
            losses_r.append(loss_r)
        total_loss_t = torch.cat(losses_t).mean()
        total_loss_r = torch.cat(losses_r).mean()
        return total_loss_t, total_loss_r

    def update_params(
        self, camera_obs_traj, proprio_obs_traj, action_traj
    ):
        camera_obs = camera_obs_traj.to(self.device)
        proprio_obs = proprio_obs_traj.to(self.device)
        action = action_traj.to(self.device)
        self.optimizer.zero_grad()
        loss_t, loss_r = self.forward(camera_obs, proprio_obs, action)
        total_loss = loss_t + loss_r
        total_loss.backward()
        self.optimizer.step()
        training_metrics = {"total loss": total_loss, "translation loss": loss_t, "rotation loss": loss_r}
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
