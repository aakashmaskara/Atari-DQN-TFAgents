#Importing necessary libraries
import tensorflow as tf
import os
import numpy as np
import warnings
import gym
from tf_agents.environments import suite_atari, tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.environments.wrappers import ActionRepeat

warnings.filterwarnings("ignore")

class RewardClippingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.clip(reward, -1.0, 1.0)

trained_games = sorted(os.listdir("checkpoints"))
print(f"Games found for resuming: {trained_games}")

for game_id in trained_games:
    print(f"\nResuming training for: {game_id}")

    env_name = f"{game_id}NoFrameskip-v4"
    checkpoint_dir = f'checkpoints/{game_id}'
    policy_dir = f'saved_policy/{game_id}'

    def make_env():
        raw_env = suite_atari.load(
            env_name,
            gym_env_wrappers=[RewardClippingWrapper] + list(suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING)
        )
        return ActionRepeat(raw_env, times=4)

    train_env = tf_py_environment.TFPyEnvironment(make_env())

    preprocessing_layer = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda obs: tf.cast(obs, tf.float32) / 255.0)
    ])

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=((32, 8, 4), (64, 4, 2), (64, 3, 1)),
        fc_layer_params=(512,)
    )

    epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=1.0,
        decay_steps=250000 // 4,
        end_learning_rate=0.01
    )

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=2.5e-4, rho=0.95, momentum=0.0, epsilon=0.00001, centered=True
    )

    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=tf.keras.losses.Huber(),
        train_step_counter=train_step_counter,
        epsilon_greedy=lambda: epsilon_fn(train_step_counter),
        target_update_period=2000,
        gamma=0.99,
        gradient_clipping=5.0
    )
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=100000
    )
    dataset = replay_buffer.as_dataset(
        sample_batch_size=128,
        num_steps=2,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)
    iterator = iter(dataset)

    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=train_step_counter
    )
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    train_checkpointer.initialize_or_restore()
    initial_step = train_step_counter.numpy()
    final_step = initial_step + 50000
    print(f"Resuming from step {initial_step}, training until {final_step}")

    train_metrics = [
        tf_metrics.AverageReturnMetric(),
        tf_metrics.AverageEpisodeLengthMetric()
    ]
    driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=4
    )

    log_interval = 1000
    eval_interval = 5000

    while train_step_counter.numpy() < final_step:
        driver.run()
        experience, _ = next(iterator)
        loss = agent.train(experience).loss
        step = train_step_counter.numpy()

        if step % log_interval == 0:
            avg_ret = train_metrics[0].result().numpy()
            avg_len = train_metrics[1].result().numpy()
            print(f"[Train:{game_id}] Step: {step}, Loss: {loss:.6f}, Avg Return: {avg_ret:.2f}, Ep Len: {avg_len:.2f}")

        if step % eval_interval == 0:
            tf_policy_saver.save(policy_dir)
            step_policy_dir = os.path.join(policy_dir, f"step_{step}")
            tf_policy_saver.save(step_policy_dir)
            train_checkpointer.save(train_step_counter)

    print(f"Completed additional 50,000 steps for {game_id}")