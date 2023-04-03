import wandb
import torch
import os
import yaml
from yaml.loader import SafeLoader
from bc_network.bcnet import Policy
from bc_network.bcnet import Policy_abs_rot
from bc_network.bcdataset import ReplayBuffer
from bc_network.bcdataset import ReplayBuffer_abs_rot 




def train_step(policy, replay_memory, config):
    for i in range(config["steps"]):
        batch = replay_memory.sample(config["batch_size"])
        camera_batch, proprio_batch, action_batch = batch
        training_metrics = policy.update_params(
            camera_batch, proprio_batch, action_batch
        )
        print("step {}".format(i))
        print("loss: {}".format(training_metrics["total loss"]))
        wandb.log(training_metrics)
    return

def train_step_exact_rotation(policy, replay_memory, config):
    for i in range(config["steps"]):
        batch = replay_memory.sample(config["batch_size"])
        camera_batch, proprio_batch, action_batch = batch
        training_metrics = policy.update_params(
            camera_batch, proprio_batch, action_batch
        )
        print("step {}".format(i))
        print("loss: {}".format(training_metrics["total loss"]))
        wandb.log(training_metrics)
    return

                

def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)
    # policy = Policy(config, device)
    policy = Policy_abs_rot(config, device)
    wandb.watch(policy, log_freq=100)
    # train_step(policy, replay_memory, config)
    torch.autograd.set_detect_anomaly(True)
    train_step_exact_rotation(policy, replay_memory, config)
    file_name = "saved_models/" + "policy.pt"
    torch.save(policy.state_dict(), file_name)
    print("Model saved to: ", file_name)
    return


if __name__ == "__main__":

    data_dir = "/home/mokhtars/Documents/bc_network/trajectories/train_overfit"
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)
        print("config: ", config)

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    # replay_memory = ReplayBuffer(config["buffer_capacity"], data_dir, config["sequence_len"])
    replay_memory = ReplayBuffer_abs_rot(config["buffer_capacity"], data_dir, config["sequence_len"])
    # Add trajectories to the replay_memory
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            replay_memory.append(subdir_path)
    wandb.init(config=config, project="vanilla_behavior_cloning")
    config = wandb.config  # important, in case the sweep gives different values
    main(config)
