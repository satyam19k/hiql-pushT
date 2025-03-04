from collections import defaultdict

import jax
import numpy as np
from tqdm import trange
import torch
import random
from preprocessor import Preprocessor
from torchvision import utils
from env.pusht.pusht_wrapper import PushTWrapper
from einops import rearrange, repeat
import imageio

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    batch,
    seeds,
    traj_len,
    idxs,
    train_dataset_dict,
    n_evals,
    filename,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        task_id: Task ID to be passed to the environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.
        eval_gaussian: Standard deviation of the Gaussian noise to add to the actions.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """

    ACTION_MEAN = torch.tensor([-0.0087, 0.0068])
    ACTION_STD = torch.tensor([0.2019, 0.2002])
    STATE_MEAN = torch.tensor([236.6155, 264.5674, 255.1307, 266.3721, 1.9584, -2.93032027,  2.54307914])
    STATE_STD = torch.tensor([101.1202, 87.0112, 52.7054, 57.4971, 1.7556, 74.84556075, 74.14009094])
    PROPRIO_MEAN = torch.tensor([236.6155, 264.5674, -2.93032027,  2.54307914])
    PROPRIO_STD = torch.tensor([101.1202, 87.0112, 74.84556075, 74.14009094])
    data_preprocessor = Preprocessor(
            action_mean=ACTION_MEAN,
            action_std=ACTION_STD,
            state_mean=STATE_MEAN,
            state_std=STATE_STD,
            proprio_mean=PROPRIO_MEAN,
            proprio_std=PROPRIO_STD,
            transform="datasets.img_transforms.default_transform",
        )
    

    actor_fn = supply_rng(agent.sample_actions, rng=jax.random.PRNGKey(np.random.randint(0, 2**32)))

    video_lengths = []
    video_starts = []


    for k in range(len(idxs)):
        current_start = idxs[k]
        for i, x in enumerate(train_dataset_dict['valids'][idxs[k]:]):
            if x == 0.0:
                segment_length = i - current_start + 1
                video_lengths.append(segment_length)
                video_starts.append(current_start)
                break;


    filtered_lengths = []
    filtered_starts = []
    for length, start in zip(video_lengths, video_starts):
        if length >= traj_len:
            filtered_lengths.append(length)
            filtered_starts.append(start)
    
    
    states = []
    actions = []
    for i in range(n_evals):
        traj_id = random.randint(0, len(filtered_starts) - 1)

        max_offset = filtered_lengths[traj_id]- traj_len

        offset = random.randint(0, max_offset)

        state = train_dataset_dict['observations'][filtered_starts[traj_id]+offset : filtered_starts[traj_id]+offset+traj_len]
        action = train_dataset_dict['actions'][filtered_starts[traj_id]+offset : filtered_starts[traj_id]+offset+traj_len-1]
        actions.append(torch.from_numpy(action))
        states.append(state)

    init_state = [x[0] for x in states]
    init_state = np.array(init_state)
    actions = torch.stack(actions)
    exec_actions = data_preprocessor.denormalize_actions(actions)

    rollout_obses_gt, rollout_states_gt = env.rollout(seeds, init_state, exec_actions.numpy())

    obs_0 = {
    key: np.expand_dims(arr[:, 0], axis=1)
    for key, arr in rollout_obses_gt.items()
}
    obs_g = {
        key: np.expand_dims(arr[:, -1], axis=1)
        for key, arr in rollout_obses_gt.items()
    }
    state_0 = init_state  # (b, d)
    state_g = rollout_states_gt[:, -1]  # (b, d)

    act=[]
    final_state = np.array([156.6379, 357.0037, 251.5347, 260.9974,   0.7817,  22.8611, -26.2061])

    for k in range(n_evals):
        in_st = init_state[k]
        res=[]
        for i in range(traj_len):
            action = actor_fn(observations=in_st, goals=final_state, temperature=eval_temperature)
            res.append(action)
        actions_model = torch.stack([torch.from_numpy(np.asarray(x)) for x in res], dim=0)
        act.append(actions_model)
    
    act = torch.stack(act)
    exec_actions_model = data_preprocessor.denormalize_actions(act)

    e_obses, e_states = env.rollout(seeds, init_state, exec_actions_model)

    eval_results = env.eval_state(state_g, e_states[:,-1])
    successes = eval_results['success']

    visual_dists = np.linalg.norm(e_obses["visual"] - obs_g["visual"], axis=1)
    mean_visual_dist = np.mean(visual_dists)

    eval_results['visual_dists'] = mean_visual_dist

    e_visuals = e_obses["visual"]

    e_visuals = data_preprocessor.transform_obs_visual(e_visuals)
    result = e_visuals.clone()  # Clone to preserve the original tensor


    e_visuals = e_visuals[: n_evals]
    goal_visual = obs_g["visual"][: n_evals]
    goal_visual = data_preprocessor.transform_obs_visual(goal_visual)

    correction = 0.3  # to distinguish env visuals and imagined visuals
    

    for idx in range(e_visuals.shape[0]):
        success_tag = "success" if successes[idx] else "failure"
        frames = []
        for i in range(e_visuals.shape[1]):
            e_obs = e_visuals[idx, i, ...]
            e_obs = torch.cat(
                [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
            )
            frame = e_obs
            frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
            frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
            frame = frame.detach().cpu().numpy()
            frames.append(frame)
        video_writer = imageio.get_writer(
            f"{filename}_{idx}_{success_tag}.mp4", fps=12
        )

        for frame in frames:
            frame = frame * 2 - 1 if frame.min() >= 0 else frame
            video_writer.append_data(
                (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
            )
        video_writer.close()

    
    e_visuals = e_visuals[:, :: 5]


    n_columns = e_visuals.shape[1]

    e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
    rollout = e_visuals
 
    n_columns += 1

    imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
    imgs_for_plotting = (
        imgs_for_plotting * 2 - 1
        if imgs_for_plotting.min() >= 0
        else imgs_for_plotting
    )
    utils.save_image(
        imgs_for_plotting,
        f"{filename}.png",
        nrow=n_columns, 
        normalize=True,
        value_range=(-1, 1),
    )


    return eval_results



