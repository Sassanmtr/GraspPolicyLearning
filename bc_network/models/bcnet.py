import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.normal import Normal



class BehaviorCloningNet(pl.LightningModule):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self._device = device
        lstm_dim = config["visual_embedding_dim"] + config["proprio_dim"]
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=2, kernel_size=3, padding=1, stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=2, out_channels=1, kernel_size=3, padding=1, stride=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2
        )
        self.lstm = nn.LSTM(lstm_dim, lstm_dim)  # , batch_first=True)
        self.linear_out = nn.Linear(lstm_dim, config["action_dim"])
        
        self.std = 0.1 * torch.ones(config["action_dim"], dtype=torch.float32)
        self.std = self.std.to(device)
        self.save_hyperparameters()
        return
    
    @property
    def device(self):
        return self._device


    def forward_step(self, camera_obs, joint_obs, lstm_state):
        vis_encoding = F.elu(self.conv1(camera_obs))
        vis_encoding = F.elu(self.conv2(vis_encoding))
        vis_encoding = F.elu(self.conv3(vis_encoding))
        vis_encoding = torch.flatten(vis_encoding, start_dim=1)
        low_dim_input = torch.cat((vis_encoding, joint_obs), dim=-1).unsqueeze(0)
        lstm_out, (h, c) = self.lstm(low_dim_input, lstm_state)
        lstm_state = (h, c)
        out = torch.tanh(self.linear_out(lstm_out))
        return out, lstm_state

    def forward(self, camera_obs_traj, joint_obs_traj, action_traj):
        losses = []
        lstm_state = None
        for idx in range(len(joint_obs_traj)):
            mu, lstm_state = self.forward_step(
                camera_obs_traj[idx], joint_obs_traj[idx], lstm_state
            )
            distribution = Normal(mu, self.std)
            log_prob = distribution.log_prob(action_traj[:, idx, :, :].permute(1, 0, 2))
            loss = -log_prob
            losses.append(loss)
        total_loss = torch.cat(losses).mean()
        return total_loss

    def training_step(self, batch, batch_idx):
        img_obs, joint_obs, action= batch
        loss = self.forward(
            img_obs, joint_obs, action
        )
        self.log_dict({"loss":loss})
        return loss

    def validation_step(self, batch, batch_idx):
        img_obs, joint_obs, action = batch
        loss = self.forward(
            img_obs, joint_obs, action
        )

        self.log_dict({"val_loss":loss})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(self.config["lr"]),
            weight_decay=float(self.config["weight_decay"]),
        )
        return optimizer

    