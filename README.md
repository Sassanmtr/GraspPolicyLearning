# Policy Learning for Real-time Generative Grasp Synthesis
Repository provides the source code for my master project in Robot Learning Lab at University of Freiburg. It consists of evaluating different imitation learning strategies for robot grasping in a challenging and realistic scenario. The experiments are performed in IsaacSim version 2022.1.1
### [Slides](https://sassanmtr.github.io/asset/MasterProject.pdf)

![teaser](figures/grasping.gif)
## Installation
 If you are using other versions of IsaacSim change the instruction accordingly. First install IsaacSim, then add to your `~/.bashrc`:
```bash
alias cd-isaac-path="cd ~/.local/share/ov/pkg/isaac_sim-2022.1.1"
alias isaac-python="~/.local/share/ov/pkg/isaac_sim-2022.1.1/python.sh"
alias source-isaac-conda="source ~/.local/share/ov/pkg/isaac_sim-2022.1.1/setup_conda_env.sh"
```
Create conda environment and install required libraries:

```bash
cd-isaac-path
conda env create -f environment.yml
conda activate isaac-sim
pip install -r requirements.txt
```

When you want to run something in a new terminal, source:

```bash
source-isaac-conda
```

## Initialization
* **Scene**: A grasping scene is designed in IsaacSim and is provided as a usd file in `bc_files/grasp_scene.usd`    
* **Robot**: An FMM(Freiburg Mobile Manipulation) robot is used for the experiments. FMM consists of a Franka arm on top of a moving base, using in Robot Learning lab of University of Freiburg. The usd file of the robot is provided at `bc_files/fmm.usd` with its customized controller at `FmmControlLite`  
* **Object**: The default object for the experiments is a simple bowl collected from the [ShapeNet dataset](https://shapenet.org/), with grasps collected from [Acronym dataset](https://github.com/NVlabs/acronym). But you can collect various objects and groundtruth grasps from these datasets and perform experiments.   

## Collect Trajectories
To collect trajectories for a default object (a simple bowl) run:

```bash
python data_collector/single_obj_data_collector.py
```

If you want to collect trajectories for your collected objects, let them in `ValidGraspObjects` directory and run:

```bash
bash data_collector/various_obj_data_collector.sh
```

## Training
* **Behavior Cloning**: For offline behavior cloning, first collect trajectories to act as an expert policy, then run:
```bash
python train.py --data_dir=/path/to/collected_data --save_dir=/path/to/save_model use_wandb=True
```
* **Interactive Imitation Learning for a single object**: For interactive imitation learning, first pr-etrain the behavior cloning agent with few number of trajectories. Then, the policy is fine-tuned while new trajectories are collecting through real-time feedback and adaptation:

```bash
python train_feedback.py --model_path=/path/to/pretrained_model --data_dir=/path/to/collected_data --save_dir=/path/to/save_model use_wandb=True
```

* **Interactive Imitation Learning for multiple objects**:
```bash
python train_feedback_multiple.py --model_path=/path/to/pretrained_model --data_dir=/path/to/collected_data --save_dir=/path/to/save_model use_wandb=True
```

## Evaluation
To evaluate the single object models, run:
```bash
python bc_eval/single_obj_validation.py --model_path=/path/to/model --log=True
```

To evaluate the multiple object models, run:
```bash
python bc_eval/multiple_objs_validation.py --model_path=/path/to/model --log=True
```

## Feedback
For any feedback or inquiries, please contact sassan.mtr@gmail.com