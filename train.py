import yaml
from yaml.loader import SafeLoader
import torch
from bc_network.datasets.bcnet_data_module import BCDataModule
from bc_network.models.bcnet import BehaviorCloningNet
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning import loggers as pl_loggers
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from functools import partial


def train(config, data_dir):
    seed_everything(42, workers=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    datamodule = BCDataModule(data_dir, config, device)
    # testloader = datamodule.test_dataloader()
    bcnet = BehaviorCloningNet(config, device)
    wb_logger = pl_loggers.WandbLogger(save_dir="./wandb_logs")
    trainer = Trainer(logger=wb_logger, deterministic=True, max_epochs=config["num_epochs"], accelerator='gpu', devices=1)
    trainer.fit(model=bcnet, datamodule=datamodule)
    # trainer.save_checkpoints()
    # trainer.test(dataloaders=datamodule)
    # trainer.test(ckpt_path="best", dataloaders=testloader) 

# def train_tune(config, epochs=200):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     datamodule = BCDataModule(data_dir, config, device)
#     bcnet = BehaviorCloningNet(config, device)
#     metrics = {"loss": "val_loss"}
#     callbacks = [TuneReportCallback(metrics, on="validation_end")] 
#     trainer = Trainer(callbacks=callbacks, deterministic=True, max_epochs=epochs, accelerator='gpu', devices=1)
#     trainer.fit(model=bcnet, datamodule=datamodule)



if __name__ == "__main__":
    data_dir = "/home/mokhtars/Documents/bc_network/bc_network/trajecories"
    with open('config.yaml') as f:
        config = yaml.load(f, Loader=SafeLoader)
        print("config: ", config)  

    # hp_config = {
    #     "visual_embedding_dim": tune.choice([2394]),
    #     "proprio_dim": tune.choice([13]),
    #     "action_dim": tune.choice([7]),
    #     "lr": tune.loguniform(1e-6, 1e-1),
    #     "weight_decay": tune.loguniform(1e-8, 1e-2),
    #     "sequence_len": tune.choice([5, 10, 15, 20, 25, 30]),
    #     "batch_size": tune.choice([2, 4, 8, 16])
    # }
    # scheduler = ASHAScheduler(
    #     max_t=10,
    #     grace_period=1,
    #     reduction_factor=2)

    # train_fn_with_parameters = tune.with_parameters(train_tune)
    # analysis = tune.run(train_fn_with_parameters,
    #     resources_per_trial={"cpu":1, "gpu":1},
    #     metric="loss",
    #     mode="min",
    #     scheduler=scheduler,
    #     config=hp_config,
    #     num_samples=2,
    #     name="asha")

    # print("Best hyperparameters found were: ", analysis.best_config)

    #-------------------------------------------------

    train(config, data_dir)

   