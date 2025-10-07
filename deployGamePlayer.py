# Importing necessary libraries
import tensorflow as tf
import numpy as np
import imageio
import os
from tf_agents.environments import suite_atari, tf_py_environment, batched_py_environment

# Configuration
policy_base_dir = "saved_policy"
video_output_dir = "videos"
MAX_EPISODE_FRAMES = 108000

# Mapping Gym environment names
env_name_map = {
    "Seaquest": "SeaquestNoFrameskip-v4",
    "SpaceInvaders": "SpaceInvadersNoFrameskip-v4"
}

os.makedirs(video_output_dir, exist_ok=True)

# Detecting available trained games
trained_games = sorted(os.listdir(policy_base_dir))
print(f"Found trained games: {trained_games}")

for game_id in trained_games:
    if game_id not in env_name_map:
        print(f"No environment mapping for game: {game_id}")
        continue

    env_name = env_name_map[game_id]
    policy_dir = os.path.join(policy_base_dir, game_id)

    # Getting latest saved step directory
    step_dirs = [d for d in os.listdir(policy_dir) if d.startswith("step_")]
    if not step_dirs:
        print(f"No saved policies found for {game_id}")
        continue
    latest_step = max(int(d.split("_")[1]) for d in step_dirs)
    best_policy_dir = os.path.join(policy_dir, f"step_{latest_step}")

    # Defining output video path
    video_filename = f"myAgentPlays-{game_id.capitalize()}.mp4"
    video_path = os.path.join(video_output_dir, video_filename)

    # Loading environment
    try:
        eval_py_env = suite_atari.load(
            env_name,
            max_episode_steps=MAX_EPISODE_FRAMES,
            gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING
        )
        eval_py_env = batched_py_environment.BatchedPyEnvironment([eval_py_env])
        time_step = eval_py_env.reset()
    except Exception as e:
        print(f"Could not load environment {env_name}: {e}")
        continue

    # Loading policy
    print(f"Loading policy from: {best_policy_dir}")
    eval_policy = tf.compat.v2.saved_model.load(best_policy_dir)

    # Recording gameplay
    print(f"Recording video for {game_id}")
    writer = imageio.get_writer(video_path, fps=30)

    frame = eval_py_env.render(mode='rgb_array')[0]
    writer.append_data(frame)

    episode_reward = 0
    frame_count = 0

    while not time_step.is_last()[0]:
        action_step = eval_policy.action(time_step)
        time_step = eval_py_env.step(action_step.action)
        frame = eval_py_env.render(mode='rgb_array')[0]
        writer.append_data(frame)
        episode_reward += time_step.reward[0]
        frame_count += 1

    writer.close()

    print(f"Game: {game_id}, Reward: {episode_reward:.2f}, Frames: {frame_count}")
    print(f"Video path: {video_path}")