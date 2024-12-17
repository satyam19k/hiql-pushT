from gymnasium.envs.registration import register

visual_dict = dict(
    ob_type='pixels',
    width=64,
    height=64,
    visualize_info=False,
)
cube_singletask_dict = dict(
    permute_blocks=False,
    reward_task_id=0,  # 0 means the default task.
)
scene_singletask_dict = dict(
    permute_blocks=False,
    reward_task_id=0,  # 0 means the default task.
)
puzzle_singletask_dict = dict(
    reward_task_id=0,  # 0 means the default task.
)

# Environments for offline goal-conditioned RL.
register(
    id='cube-single-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='single',
    ),
)
register(
    id='visual-cube-single-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='single',
        **visual_dict,
    ),
)
register(
    id='cube-double-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='double',
    ),
)
register(
    id='visual-cube-double-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='double',
        **visual_dict,
    ),
)
register(
    id='cube-triple-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='triple',
    ),
)
register(
    id='visual-cube-triple-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='triple',
        **visual_dict,
    ),
)
register(
    id='cube-quadruple-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='quadruple',
    ),
)
register(
    id='visual-cube-quadruple-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='quadruple',
        **visual_dict,
    ),
)

register(
    id='scene-v0',
    entry_point='ogbench.manipspace.envs.scene_env:SceneEnv',
    max_episode_steps=750,
    kwargs=dict(
        env_type='scene',
    ),
)
register(
    id='visual-scene-v0',
    entry_point='ogbench.manipspace.envs.scene_env:SceneEnv',
    max_episode_steps=750,
    kwargs=dict(
        env_type='scene',
        **visual_dict,
    ),
)

register(
    id='puzzle-3x3-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='3x3',
    ),
)
register(
    id='visual-puzzle-3x3-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='3x3',
        **visual_dict,
    ),
)
register(
    id='puzzle-4x4-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='4x4',
    ),
)
register(
    id='visual-puzzle-4x4-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='4x4',
        **visual_dict,
    ),
)
register(
    id='puzzle-4x5-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x5',
    ),
)
register(
    id='visual-puzzle-4x5-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x5',
        **visual_dict,
    ),
)
register(
    id='puzzle-4x6-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x6',
    ),
)
register(
    id='visual-puzzle-4x6-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x6',
        **visual_dict,
    ),
)

# Environments for reward-based single-task offline RL.
register(
    id='cube-single-singletask-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='single',
        **cube_singletask_dict,
    ),
)
register(
    id='visual-cube-single-singletask-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=200,
    kwargs=dict(
        env_type='single',
        **visual_dict,
        **cube_singletask_dict,
    ),
)
register(
    id='cube-double-singletask-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='double',
        **cube_singletask_dict,
    ),
)
register(
    id='visual-cube-double-singletask-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='double',
        **visual_dict,
        **cube_singletask_dict,
    ),
)
register(
    id='cube-triple-singletask-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='triple',
        **cube_singletask_dict,
    ),
)
register(
    id='visual-cube-triple-singletask-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='triple',
        **visual_dict,
        **cube_singletask_dict,
    ),
)
register(
    id='cube-quadruple-singletask-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='quadruple',
        **cube_singletask_dict,
    ),
)
register(
    id='visual-cube-quadruple-singletask-v0',
    entry_point='ogbench.manipspace.envs.cube_env:CubeEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='quadruple',
        **visual_dict,
        **cube_singletask_dict,
    ),
)

register(
    id='scene-singletask-v0',
    entry_point='ogbench.manipspace.envs.scene_env:SceneEnv',
    max_episode_steps=750,
    kwargs=dict(
        env_type='scene',
        **scene_singletask_dict,
    ),
)
register(
    id='visual-scene-singletask-v0',
    entry_point='ogbench.manipspace.envs.scene_env:SceneEnv',
    max_episode_steps=750,
    kwargs=dict(
        env_type='scene',
        **visual_dict,
        **scene_singletask_dict,
    ),
)

register(
    id='puzzle-3x3-singletask-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='3x3',
        **puzzle_singletask_dict,
    ),
)
register(
    id='visual-puzzle-3x3-singletask-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='3x3',
        **visual_dict,
        **puzzle_singletask_dict,
    ),
)
register(
    id='puzzle-4x4-singletask-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='4x4',
        **puzzle_singletask_dict,
    ),
)
register(
    id='visual-puzzle-4x4-singletask-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=500,
    kwargs=dict(
        env_type='4x4',
        **visual_dict,
        **puzzle_singletask_dict,
    ),
)
register(
    id='puzzle-4x5-singletask-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x5',
        **puzzle_singletask_dict,
    ),
)
register(
    id='visual-puzzle-4x5-singletask-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x5',
        **visual_dict,
        **puzzle_singletask_dict,
    ),
)
register(
    id='puzzle-4x6-singletask-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x6',
        **puzzle_singletask_dict,
    ),
)
register(
    id='visual-puzzle-4x6-singletask-v0',
    entry_point='ogbench.manipspace.envs.puzzle_env:PuzzleEnv',
    max_episode_steps=1000,
    kwargs=dict(
        env_type='4x6',
        **visual_dict,
        **puzzle_singletask_dict,
    ),
)
