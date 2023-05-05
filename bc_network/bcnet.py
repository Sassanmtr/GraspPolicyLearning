import torch
import roma
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torchvision.models import resnet18


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


class Policy_twist(nn.Module):
    def __init__(self, config, device):
        super(Policy_twist, self).__init__()
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
        self.loss = nn.MSELoss().to(device)
        self.binary_loss = nn.BCEWithLogitsLoss().to(device)
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
            losses_t.append(self.loss(output[:,:,:3], action_traj[idx][:,:,:3].permute(1, 0, 2)))
            losses_r.append(self.loss(output[:,:,3:-1], action_traj[idx][:,:,3:-1].permute(1, 0, 2)))
            losses_g.append(self.binary_loss(output[:,:,-1], action_traj[idx][:,:,-1].permute(1, 0)))

        total_loss_t = sum(losses_t)
        total_loss_r = sum(losses_r)
        total_loss_g = sum(losses_g)

        return total_loss_t.float(), total_loss_r.float(), total_loss_g.float()

    def update_params(
        self, camera_obs_traj, proprio_obs_traj, action_traj
    ):
        camera_obs = camera_obs_traj.to(self.device).float()
        proprio_obs = proprio_obs_traj.to(self.device).float()
        action = action_traj.to(self.device).float()
        self.optimizer.zero_grad()
        loss_t, loss_r , loss_g= self.forward(camera_obs, proprio_obs, action)
        total_loss = loss_t + loss_r #+ (loss_g/1)
        total_loss.backward()
        self.optimizer.step()
        training_metrics = {"total loss": total_loss, "translation loss": loss_t, "rotation loss": loss_r}#, "gripper loss": loss_g}
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

# class ResNet18Conv(ConvBase):
#     """
#     A ResNet18 block that can be used to process input images.
#     """
#     def __init__(
#         self,
#         input_channel=3,
#         pretrained=False,
#         input_coord_conv=False,
#     ):
#         """
#         Args:
#             input_channel (int): number of input channels for input images to the network.
#                 If not equal to 3, modifies first conv layer in ResNet to handle the number
#                 of input channels.
#             pretrained (bool): if True, load pretrained weights for all ResNet layers.
#             input_coord_conv (bool): if True, use a coordinate convolution for the first layer
#                 (a convolution where input channels are modified to encode spatial pixel location)
#         """
#         super(ResNet18Conv, self).__init__()
#         net = vision_models.resnet18(pretrained=pretrained)

#         if input_coord_conv:
#             net.conv1 = CoordConv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         elif input_channel != 3:
#             net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

#         # cut the last fc layer
#         self._input_coord_conv = input_coord_conv
#         self._input_channel = input_channel
#         self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

#     def output_shape(self, input_shape):
#         """
#         Function to compute output shape from inputs to this module. 

#         Args:
#             input_shape (iterable of int): shape of input. Does not include batch dimension.
#                 Some modules may not need this argument, if their output does not depend 
#                 on the size of the input, or if they assume fixed size input.

#         Returns:
#             out_shape ([int]): list of integers corresponding to output shape
#         """
#         assert(len(input_shape) == 3)
#         out_h = int(math.ceil(input_shape[1] / 32.))
#         out_w = int(math.ceil(input_shape[2] / 32.))
#         return [512, out_h, out_w]

#     def __repr__(self):
#         """Pretty print network."""
#         header = '{}'.format(str(self.__class__.__name__))
#         return header + '(input_channel={}, input_coord_conv={})'.format(self._input_channel, self._input_coord_conv)



class ResNet_Policy_twist_backup(nn.Module):
    def __init__(self, config, device):
        super(ResNet_Policy_twist, self).__init__()
        self._device = device
        lstm_dim1 = config["resnet_visual_embedding_dim"] + config["proprio_dim"]
        lstm_dim2 = config["resnet_visual_embedding_dim"] + config["proprio_dim"]
        # Load a pre-trained ResNet18 model and remove the last layer
        self.resnet = resnet18(pretrained=True).to(device)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(output_size=(128, 256))  # set output size to (300, 500)
        # conv layers
        self.conv1 = nn.Conv2d(
            in_channels=512, out_channels=3, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.conv2 = nn.Conv2d(
            in_channels=3, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.conv3 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)
        # lstm layers
        self.lstm1 = nn.LSTM(lstm_dim1, lstm_dim1).to(device)
        self.linear_out = nn.Linear(lstm_dim1, config["action_dim"]).to(device)
        self.lstm2 = nn.LSTM(lstm_dim1, lstm_dim2).to(device)
        self.linear_out2 = nn.Linear(lstm_dim2, config["action_dim"]).to(device)
        # optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.loss = nn.MSELoss().to(device)
        self.binary_loss = nn.BCEWithLogitsLoss().to(device)
        return

    @property
    def device(self):
        return self._device

    def forward_step(self, camera_obs, proprio_obs, lstm_state1, lstm_state2):
        vis_encoding = self.resnet(camera_obs)
        vis_encoding = self.pool(vis_encoding)
        vis_encoding = F.elu(self.conv1(vis_encoding))
        vis_encoding = F.elu(self.conv2(vis_encoding))
        vis_encoding = F.elu(self.conv3(vis_encoding))
        vis_encoding = torch.flatten(vis_encoding, start_dim=1)
        low_dim_input = torch.cat((vis_encoding, proprio_obs), dim=-1).unsqueeze(0)
        lstm_out1, (h1, c1) = self.lstm1(low_dim_input, lstm_state1)
        lstm_out2, (h2, c2) = self.lstm2(lstm_out1, lstm_state2)
        lstm_state1 = (h1, c1)
        lstm_state2 = (h2, c2)
        out = torch.tanh(self.linear_out(lstm_out2))
        return out, lstm_state1, lstm_state2

    def forward(self, camera_obs_traj, proprio_obs_traj, action_traj):
        losses_t = []
        losses_r = []
        losses_g = []
        lstm_state1 = None
        lstm_state2 = None
        for idx in range(len(proprio_obs_traj)):
            output, lstm_state1, lstm_state2 = self.forward_step(
                camera_obs_traj[idx], proprio_obs_traj[idx], lstm_state1, lstm_state2
            )
            losses_t.append(self.loss(output[:,:,:3], action_traj[idx][:,:,:3].permute(1, 0, 2)))
            losses_r.append(self.loss(output[:,:,3:-1], action_traj[idx][:,:,3:-1].permute(1, 0, 2)))
            losses_g.append(self.binary_loss(output[:,:,-1], action_traj[idx][:,:,-1].permute(1, 0)))

        total_loss_t = sum(losses_t)
        total_loss_r = sum(losses_r)
        total_loss_g = sum(losses_g)

        return total_loss_t.float(), total_loss_r.float(), total_loss_g.float()

    def update_params(
        self, camera_obs_traj, proprio_obs_traj, action_traj
    ):
        camera_obs = camera_obs_traj.to(self.device).float()
        proprio_obs = proprio_obs_traj.to(self.device).float()
        action = action_traj.to(self.device).float()
        self.optimizer.zero_grad()
        loss_t, loss_r , loss_g= self.forward(camera_obs, proprio_obs, action)
        total_loss = loss_t + loss_r #+ (loss_g/1)
        total_loss.backward()
        self.optimizer.step()
        training_metrics = {"total loss": total_loss, "translation loss": loss_t, "rotation loss": loss_r, "gripper loss": loss_g}
        return training_metrics

    def predict(self, camera_obs, proprio_obs, lstm_state1, lstm_state2):
        camera_obs_th = torch.tensor(camera_obs, dtype=torch.float32).unsqueeze(0)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32).unsqueeze(0)
        camera_obs_th = camera_obs_th.to(self.device)
        proprio_obs_th = proprio_obs_th.to(self.device)
        with torch.no_grad():
            action_th, lstm_state1, lstm_state2 = self.forward_step(
                camera_obs_th, proprio_obs_th, lstm_state1, lstm_state2
            )
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state1, lstm_state2




class ResNet_Policy_twist(nn.Module):
    def __init__(self, config, device):
        super(ResNet_Policy_twist, self).__init__()
        self._device = device
        lstm_dim1 = config["resnet_visual_embedding_dim"] + config["proprio_dim"]
        lstm_dim2 = config["resnet_visual_embedding_dim"] + config["proprio_dim"]
        # Load a pre-trained ResNet18 model and remove the last layer
        self.resnet_rgb = resnet18(pretrained=True).to(device)
        self.resnet_rgb = nn.Sequential(*list(self.resnet_rgb.children())[:-2])
        # self.pool_rgb = nn.AdaptiveAvgPool2d(output_size=(128, 256))  # set output size to (300, 500)
        self.resnet_depth = resnet18(pretrained=True).to(device)
        self.resnet_depth.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
        self.resnet_depth = nn.Sequential(*list(self.resnet_depth.children())[:-2])
        # self.pool_depth = nn.AdaptiveAvgPool2d(output_size=(128, 256))  # set output size to (300, 500)
        # conv layers
        self.conv_rgb1 = nn.Conv2d(
            in_channels=512, out_channels=3, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.conv_rgb2 = nn.Conv2d(
            in_channels=3, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.conv_rgb3 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)

        self.conv_depth1 = nn.Conv2d(
            in_channels=512, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.conv_depth2 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)
        self.conv_depth3 = nn.Conv2d(
            in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2
        ).to(device)
        # lstm layers
        self.lstm1 = nn.LSTM(lstm_dim1, lstm_dim1).to(device)
        self.linear_out = nn.Linear(lstm_dim1, config["action_dim"]).to(device)
        self.lstm2 = nn.LSTM(lstm_dim1, lstm_dim2).to(device)
        self.linear_out2 = nn.Linear(lstm_dim2, config["action_dim"]).to(device)
        # optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.loss = nn.MSELoss().to(device)
        self.binary_loss = nn.BCEWithLogitsLoss().to(device)
        return

    @property
    def device(self):
        return self._device

    def forward_step(self, rgb_obs, depth_obs, proprio_obs, lstm_state1, lstm_state2):
        #rgb pass
        vis_encoding_rgb = self.resnet_rgb(rgb_obs)
        vis_encoding_rgb = F.elu(self.conv_rgb1(vis_encoding_rgb))
        vis_encoding_rgb = F.elu(self.conv_rgb2(vis_encoding_rgb))
        vis_encoding_rgb = F.elu(self.conv_rgb3(vis_encoding_rgb))
        vis_encoding_rgb = torch.flatten(vis_encoding_rgb, start_dim=1)
        #depth pass
        vis_encoding_depth = self.resnet_depth(depth_obs)
        vis_encoding_depth = F.elu(self.conv_depth1(vis_encoding_depth))
        vis_encoding_depth = F.elu(self.conv_depth2(vis_encoding_depth))
        vis_encoding_depth = F.elu(self.conv_depth3(vis_encoding_depth))
        vis_encoding_depth = torch.flatten(vis_encoding_depth, start_dim=1)
        low_dim_input = torch.cat((vis_encoding_rgb, vis_encoding_depth, proprio_obs), dim=-1).unsqueeze(0)
        lstm_out1, (h1, c1) = self.lstm1(low_dim_input, lstm_state1)
        lstm_out2, (h2, c2) = self.lstm2(lstm_out1, lstm_state2)
        lstm_state1 = (h1, c1)
        lstm_state2 = (h2, c2)
        out = torch.tanh(self.linear_out(lstm_out2))
        return out, lstm_state1, lstm_state2

    def forward(self, rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj):
        losses_t = []
        losses_r = []
        losses_g = []
        lstm_state1 = None
        lstm_state2 = None
        for idx in range(len(proprio_obs_traj)):
            output, lstm_state1, lstm_state2 = self.forward_step(
                rgb_obs_traj[idx], depth_obs_traj[idx],proprio_obs_traj[idx], lstm_state1, lstm_state2
            )
            losses_t.append(self.loss(output[:,:,:3], action_traj[idx][:,:,:3].permute(1, 0, 2)))
            losses_r.append(self.loss(output[:,:,3:-1], action_traj[idx][:,:,3:-1].permute(1, 0, 2)))
            losses_g.append(self.binary_loss(output[:,:,-1], action_traj[idx][:,:,-1].permute(1, 0)))

        total_loss_t = sum(losses_t)
        total_loss_r = sum(losses_r)
        total_loss_g = sum(losses_g)

        return total_loss_t.float(), total_loss_r.float(), total_loss_g.float()

    def update_params(
        self, rgb_obs_traj, depth_obs_traj, proprio_obs_traj, action_traj
    ):
        rgb_obs = rgb_obs_traj.to(self.device).float()
        depth_obs = depth_obs_traj.to(self.device).float()
        proprio_obs = proprio_obs_traj.to(self.device).float()
        action = action_traj.to(self.device).float()
        self.optimizer.zero_grad()
        loss_t, loss_r , loss_g= self.forward(rgb_obs, depth_obs ,proprio_obs, action)
        total_loss = loss_t + loss_r #+ (loss_g/1)
        total_loss.backward()
        self.optimizer.step()
        training_metrics = {"total loss": total_loss, "translation loss": loss_t, "rotation loss": loss_r, "gripper loss": loss_g}
        return training_metrics

    def predict(self, rgb_obs, depth_obs, proprio_obs, lstm_state1, lstm_state2):
        rgb_obs_th = torch.tensor(rgb_obs, dtype=torch.float32).unsqueeze(0)
        depth_obs_th = torch.tensor(depth_obs, dtype=torch.float32).unsqueeze(0)
        proprio_obs_th = torch.tensor(proprio_obs, dtype=torch.float32).unsqueeze(0)
        rgb_obs_th = rgb_obs_th.to(self.device)
        depth_obs_th = depth_obs_th.to(self.device)
        proprio_obs_th = proprio_obs_th.to(self.device)

        with torch.no_grad():
            action_th, lstm_state1, lstm_state2 = self.forward_step(
                rgb_obs_th.permute(0, 3, 1, 2), depth_obs_th.permute(0, 3, 1, 2), proprio_obs_th, lstm_state1, lstm_state2
            )
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
            action[-1] = binary_gripper(action[-1])
        return action, lstm_state1, lstm_state2


def binary_gripper(gripper_action):
    if gripper_action >= 0.0:
        gripper_action = 0.8
    elif gripper_action < 0.0:
        gripper_action = 0.0
    return gripper_action
