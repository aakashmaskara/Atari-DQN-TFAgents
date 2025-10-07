# Importing necessary libraries
import os
import numpy as np
import imageio
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.environments import suite_atari, tf_py_environment

# Configuration
BASE_POLICY_DIR = 'saved_policy'
VIDEO_OUTPUT_DIR = 'videos'
EVAL_EPISODES = 5
MAX_FRAMES_PER_EPISODE = 10000

# Game-to-environment mapping
env_name_map = {
    'Seaquest': 'SeaquestNoFrameskip-v4',
    'SpaceInvaders': 'SpaceInvadersNoFrameskip-v4'
}

os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
games = sorted(os.listdir(BASE_POLICY_DIR))

for game_id in games:
    print(f"\nGame: {game_id}")
    policy_base_dir = os.path.join(BASE_POLICY_DIR, game_id)
    checkpoint_steps = sorted([
        int(d.split('_')[1]) for d in os.listdir(policy_base_dir)
        if d.startswith('step_') and os.path.isdir(os.path.join(policy_base_dir, d))
    ])

    if not checkpoint_steps:
        print(f"No checkpoint folders found in {policy_base_dir}")
        continue

    if game_id not in env_name_map:
        print(f"No environment mapping found for {game_id}")
        continue

    env_name = env_name_map[game_id]
    try:
        eval_py_env = suite_atari.load(
            env_name,
            max_episode_steps=MAX_FRAMES_PER_EPISODE,
            gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING
        )
        eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    except Exception as e:
        print(f"Could not load environment {env_name}: {e}")
        continue

    returns = []

    for step in checkpoint_steps:
        policy_path = os.path.join(policy_base_dir, f"step_{step}")
        if not os.path.exists(policy_path):
            print(f"Policy not found at {policy_path}")
            returns.append(0)
            continue

        print(f"Step {step}: Loading policy from {policy_path}")
        saved_policy = tf.compat.v2.saved_model.load(policy_path)
        avg_return = 0.0

        for _ in range(EVAL_EPISODES):
            time_step = eval_env.reset()
            episode_return = 0.0
            while not time_step.is_last():
                action_step = saved_policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                episode_return += float(time_step.reward)
            avg_return += episode_return

        avg_return /= EVAL_EPISODES
        returns.append(avg_return)
        print(f"Step {step}: Avg Return = {avg_return:.2f}")

    # Plotting
    curve_output_file = f'trainingCurve-{game_id.capitalize()}.png'
    plt.figure()
    plt.plot(checkpoint_steps, returns, marker='o')
    plt.title(f"Average Return vs Steps for {game_id.capitalize()}")
    plt.xlabel("Training Steps")
    plt.ylabel("Average Return")
    plt.grid()
    plt.savefig(curve_output_file)
    plt.close()
    print(f"Curve plot saved: {curve_output_file}")

    # Selecting video episodes: poor, intermediate, best
    returns_np = np.array(returns)
    poor_idx = np.argmin(returns_np)
    best_idx = np.argmax(returns_np)
    intermediate_idx = len(returns_np) // 2

    selected = {
        'poor': checkpoint_steps[poor_idx],
        'intermediate': checkpoint_steps[intermediate_idx],
        'best': checkpoint_steps[best_idx]
    }

    def record_policy(step, filename):
        print(f"Recording for Step {step}")
        policy_path = os.path.join(policy_base_dir, f'step_{step}')
        saved_policy = tf.compat.v2.saved_model.load(policy_path)

        frames = []
        time_step = eval_env.reset()
        frames.append(eval_env.pyenv.envs[0].render(mode='rgb_array'))

        while not time_step.is_last():
            action_step = saved_policy.action(time_step)
            time_step = eval_env.step(action_step.action)
            frame = eval_env.pyenv.envs[0].render(mode='rgb_array')
            frames.append(frame)

            if len(frames) > MAX_FRAMES_PER_EPISODE:
                break

        video_path = os.path.join(VIDEO_OUTPUT_DIR, filename)
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Saved video: {video_path}")

    # Generating videos
    for quality, step in selected.items():
        filename = f"myAgentPlays-{game_id.capitalize()}-{quality}.mp4"
        record_policy(step, filename)