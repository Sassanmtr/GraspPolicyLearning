import argparse
import os
from pathlib import Path
import yaml
from yaml.loader import SafeLoader
import torch
import wandb
from bc_network.networks import PolicyEuler, PolicyAbsRot, PolicyTwist, PolicyBPTT
from bc_network.datasets import ReplayBufferEuler, ReplayBufferAbsRot, ReplayBufferTwist, BPTTReplayBufferTwist


def get_policy(policy_name, config, device):
    policies = {
        'PolicyEuler': PolicyEuler,
        'PolicyAbsRot': PolicyAbsRot,
        'PolicyTwist': PolicyTwist,
        'PolicyBPTT': PolicyBPTT,
    }
    return policies[policy_name](config, device)


def get_replay_buffer(buffer_name, config, data_dir):
    replay_buffers = {
        'BPTTReplayBufferTwist': BPTTReplayBufferTwist,
        'ReplayBufferAbsRot': ReplayBufferAbsRot,
        'ReplayBufferTwist': ReplayBufferTwist,
        'ReplayBufferEuler': ReplayBufferEuler,
    }
    return replay_buffers[buffer_name](config["buffer_capacity"], data_dir)


def train_step(policy, replay_memory, config, save_path):
    save_interval = config.get("save_interval", 1000)
    for i in range(config["steps"]):
        batch = replay_memory.padded_sample(config["batch_size"], 500)
        rgb_batch, depth_batch, proprio_batch, action_batch = batch
        training_metrics = policy.update_params(rgb_batch, depth_batch, proprio_batch, action_batch)
        print(f"step {i} loss: {training_metrics['total loss']}")
        wandb.log(training_metrics)
        if i % save_interval == 0:
            file_name = os.path.join(save_path, f"policy_{policy.__class__.__name__}_step_{i}.pt")
            torch.save(policy.state_dict(), file_name)
            print(f"Model saved to: {file_name}")
    return


def main(config, data_dir, model_dir, use_wandb):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    replay_memory = get_replay_buffer(config["replay_buffer"], config, data_dir)
    for subdir in os.listdir(data_dir):
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.isdir(subdir_path):
            replay_memory.append(subdir_path)
    
    policy = get_policy(config["policy"], config, device)
    
    if use_wandb:
        wandb.init(config=config, project="BC Grasping")
        wandb.watch(policy, log_freq=100)
    
    train_step(policy, replay_memory, config, model_dir)
    if use_wandb:
        wandb.finish()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a policy model for robotic grasping.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save the models.")
    parser.add_argument("--use_wandb", action='store_true', help="Whether to use WandB for logging.")

    args = parser.parse_args()
    config_path = Path.cwd() / "config.yaml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=SafeLoader)
        print("Config:", config)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    main(config, args.data_dir, args.save_dir, args.use_wandb)