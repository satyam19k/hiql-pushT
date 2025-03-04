import json
import os
import random
import time
from collections import defaultdict
import torch
import pickle
from flax.core.frozen_dict import freeze
import jax
import numpy as np
import tqdm
import wandb
from absl import app, flags
from agents import agents
from ml_collections import config_flags
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb


import gym
import json
import torch
import numpy as np
from env.venv import SubprocVectorEnv


FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_integer('eval_tasks', None, 'Number of tasks to evaluate (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Number of episodes for each task.')
flags.DEFINE_float('eval_temperature', 0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Action Gaussian noise for evaluation.')
flags.DEFINE_integer('video_episodes', 1, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to evaluate on CPU.')

# config_flags.DEFINE_config_file('agent', 'agents/gciql.py', lock_config=False)
config_flags.DEFINE_config_file('agent', 'agents/hiql.py', lock_config=False)

n_evals = 10
eval_seed = [99* n + 1 for n in range(n_evals)]
filename = "/Users/jayesh/MSCS/RL/HIQL_pushT/ogbench_prev/impls/plots/rollout"


def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)


    def prepare_dataset(data_path,dset_type):
        data_path = data_path + dset_type

        ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
        ACTION_STD = torch.tensor([0.2019, 0.2002])

        action_scale=100.0

        with open(data_path + "seq_lengths.pkl" ,"rb") as f:
            seq_lengths = pickle.load(f)

        states = torch.load(data_path + "states.pth")
        vel = torch.load(data_path + "velocities.pth")
        actions = torch.load(data_path + "rel_actions.pth")


        states = states.float()
        actions = actions.float()
        vel = vel.float()


        states_0=states[0,:seq_lengths[0],:] 
        vel_0 = vel[0,:seq_lengths[0],:]
        actions_0 = actions[0,:seq_lengths[0],:]
        actions_0 = actions_0 / action_scale
        

        observations = torch.cat([states_0.float(), vel_0.float()], dim=-1)
        actions_dt = (actions_0 - ACTION_MEAN) / ACTION_STD

        terminals = torch.zeros(observations.shape[0], dtype=torch.float32)
        valids = torch.ones(observations.shape[0], dtype=torch.float32)
        terminals[-2:] = 1.0
        valids[-1] = 0.0


        for i in range(1,states.shape[0]):
            seq_len = seq_lengths[i]

            states_0=states[0,:seq_len,:] 
            vel_0 = vel[0,:seq_len,:]

            
            actions_0 = actions[0,:seq_len,:]
            actions_0 = actions_0 / action_scale
            

            observations_i = torch.cat([states_0.float(), vel_0.float()], dim=-1)
            actions_i = (actions_0 - ACTION_MEAN) / ACTION_STD

            observations = torch.cat((observations, observations_i), dim=0)
            actions_dt = torch.cat((actions_dt, actions_i), dim=0)

            terminals_i = torch.zeros(observations_i.shape[0], dtype=torch.float32)
            valids_i = torch.ones(observations_i.shape[0], dtype=torch.float32)
            terminals_i[-2:] = 1.0
            valids_i[-1] = 0.0

            terminals = torch.cat((terminals, terminals_i), dim=0)
            valids = torch.cat((valids, valids_i), dim=0)
        
        print(observations.shape)
        print(actions_dt.shape)

        data_dict = {
            'terminals': terminals.detach().cpu().numpy(),
            'valids': valids.detach().cpu().numpy(),
            'observations': observations.detach().cpu().numpy(),
            'actions': actions_dt.detach().cpu().numpy()
        }
        return freeze(data_dict)



    data_path = "/Users/jayesh/MSCS/RL/HIQL_pushT/pusht_noise/"

    train_dataset_dict = prepare_dataset(data_path,"train/")
    val_dataset_dict = prepare_dataset(data_path,"val/")


    config = FLAGS.agent

    dataset_class = {
        'GCDataset': GCDataset,
        'HGCDataset': HGCDataset,
    }[config['dataset_class']]
    train_dataset = dataset_class(Dataset.create(**train_dataset_dict), config)
    if val_dataset_dict is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset_dict), config)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch,_ = train_dataset.sample(1)


    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        # Update agent.
        batch,idxs = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if i == 1 or i % FLAGS.eval_interval == 0:
            if FLAGS.eval_on_cpu:
                eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
            else:
                eval_agent = agent
            eval_metrics = {}
            overall_metrics = defaultdict(list)
            env = SubprocVectorEnv(
                [
                    lambda: gym.make(
                        "pusht", [], {'with_velocity': True, 'with_target': True}
                    )
                    for _ in range(n_evals)
                ]
            )

            eval_results = evaluate(
                agent=eval_agent,
                env=env,
                traj_len=26,
                idxs=idxs,
                train_dataset_dict=train_dataset_dict,
                n_evals=n_evals,
                filename=filename,
                config=config,
                batch = batch,
                seeds=eval_seed,
                eval_temperature=FLAGS.eval_temperature,
                eval_gaussian=FLAGS.eval_gaussian,
            )
            # renders.extend(cur_renders)
            metric_names = ['success']
            eval_metrics.update(
                {f'evaluation/{"test_eval"}_{k}': v for k, v in eval_results.items()}
            )

            for k, v in eval_results.items():
                if k in metric_names:
                    overall_metrics[k].append(v)

            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)
            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent.
        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    app.run(main)
