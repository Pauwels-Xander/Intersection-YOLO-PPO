import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import os
import sys
import matplotlib.pyplot as plt


class TrafficIntersectionEnv(gym.Env):
    """
    Custom Gymnasium environment for a four-lane intersection traffic signal.
    Observation: 8-dimensional (4 normalized queue lengths, 4 normalized wait times).
    Action: Discrete 4 (which lane gets green light).
    """

    def __init__(self):
        super().__init__()
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        # Simulation parameters
        self.step_time = 10  # seconds per step
        self.max_steps = 8640  # 24 hours / 10s = 8640 steps per episode
        # Poisson arrival rates
        self.lambda_peak = 0.5  # expected cars per 10s in peak hours (~180 per hour)
        self.lambda_offpeak = 0.2  # expected cars per 10s in off-peak
        # Normalization constants for observation
        self.max_queue_norm = 20.0  # normalize queue length by 20 vehicles
        self.max_wait_time_norm = 300.0  # normalize average wait by 300 sec (5 minutes)
        # Internal state
        self.step_count = None
        self.queues = None  # list of lists for each lane's waiting times
        self.last_action = None  # last lane that was green
        self.total_wait = None  # total waiting time of all vehicles currently in system
        # Seed the RNG for reproducibility (optional)
        self.np_random, _ = gym.utils.seeding.np_random(None)

    def reset(self, seed=None, options=None):
        # Gymnasium API: can accept a seed here
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Initialize state
        self.step_count = 0
        self.last_action = None
        self.queues = [[] for _ in range(4)]
        self.total_wait = 0.0
        obs = np.zeros(8, dtype=np.float32)
        return obs, {}

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action!"
        switching = (self.last_action is not None and action != self.last_action)

        # 1. Vehicle arrivals for this step (Poisson random for each lane)
        hour = (self.step_count * self.step_time) / 3600.0
        if (8 <= hour < 10) or (16 <= hour < 18):
            lambda_current = self.lambda_peak
        else:
            lambda_current = self.lambda_offpeak
        arrivals = [self.np_random.poisson(lambda_current) for _ in range(4)]

        # 2. Update waiting times for existing vehicles
        for i in range(4):
            if len(self.queues[i]) > 0:
                for j in range(len(self.queues[i])):
                    self.queues[i][j] += self.step_time
                self.total_wait += len(self.queues[i]) * self.step_time
            if arrivals[i] > 0:
                self.queues[i].extend([0.0] * arrivals[i])

        # 3. Service vehicles from the selected lane (or not, if switching)
        if switching:
            served_count = 0
            served_wait_total = 0.0
        else:
            queue = self.queues[action]
            served_count = min(len(queue), 5)
            served_wait_total = 0.0
            for _ in range(served_count):
                waited_time = queue.pop(0)
                served_wait_total += waited_time
            self.total_wait -= served_wait_total

        self.last_action = action

        # 4. Calculate reward
        reward = - self.total_wait

        # 5. Build observation for next state
        obs = np.zeros(8, dtype=np.float32)
        for i in range(4):
            queue_len = len(self.queues[i])
            queue_norm = min(queue_len / self.max_queue_norm, 1.0)
            avg_wait = sum(self.queues[i]) / queue_len if queue_len > 0 else 0.0
            wait_norm = min(avg_wait / self.max_wait_time_norm, 1.0)
            obs[i] = queue_norm
            obs[4 + i] = wait_norm

        self.step_count += 1
        terminated = False
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "switching": switching,
            "served_cars": served_count,
            "served_wait_time": served_wait_total
        }
        return obs, reward, terminated, truncated, info


# Custom callback to print progress every 10,000 steps
class PrintTimestepsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_print >= 10000:
            print(f"Training timesteps: {self.num_timesteps}")
            self.last_print = self.num_timesteps // 10000 * 10000
        return True


# Baseline policy: Fixed-time cycling through actions with a fixed green duration
def evaluate_fixed_time(env, green_duration_steps=600):
    obs, _ = env.reset()
    total_reward = 0.0
    current_action = 0
    steps_this_phase = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(current_action)
        total_reward += reward
        steps_this_phase += 1
        if steps_this_phase >= green_duration_steps:
            current_action = (current_action + 1) % 4
            steps_this_phase = 0
        if terminated or truncated:
            break
    return total_reward


# Baseline policy: Random actions
def evaluate_random(env):
    obs, _ = env.reset()
    total_reward = 0.0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


# Evaluate a given trained model (PPO policy) for one episode
def evaluate_trained_policy(env, model):
    obs, _ = env.reset()
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


# New functions that record queue lengths over time during an episode
def evaluate_fixed_time_and_record(env, green_duration_steps=30):
    obs, _ = env.reset()
    total_reward = 0.0
    queue_record = []
    current_action = 0
    steps_this_phase = 0
    while True:
        # Record the total queue length across all lanes
        total_queue = sum(len(q) for q in env.queues)
        queue_record.append(total_queue)
        obs, reward, terminated, truncated, info = env.step(current_action)
        total_reward += reward
        steps_this_phase += 1
        if steps_this_phase >= green_duration_steps:
            current_action = (current_action + 1) % 4
            steps_this_phase = 0
        if terminated or truncated:
            break
    return total_reward, queue_record


def evaluate_random_and_record(env):
    obs, _ = env.reset()
    total_reward = 0.0
    queue_record = []
    while True:
        total_queue = sum(len(q) for q in env.queues)
        queue_record.append(total_queue)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward, queue_record


def evaluate_trained_policy_and_record(env, model):
    obs, _ = env.reset()
    total_reward = 0.0
    queue_record = []
    while True:
        total_queue = sum(len(q) for q in env.queues)
        queue_record.append(total_queue)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward, queue_record


# Main execution: train or evaluate based on command-line argument
if __name__ == "__main__":


    mode = "eval" ###################################################################################################################################################################


    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    if mode == "train":
        num_envs = 4

        def make_env():
            def _init():
                env = TrafficIntersectionEnv()
                return env
            return _init

        env_fns = [make_env() for _ in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns)
        eval_env = TrafficIntersectionEnv()
        eval_callback = EvalCallback(eval_env, eval_freq=10000, n_eval_episodes=1, verbose=0)
        print_callback = PrintTimestepsCallback()
        model = PPO("MlpPolicy", vec_env, verbose=1, device='cpu')
        model.learn(total_timesteps=800000, callback=[print_callback, eval_callback])
        model.save("ppo_traffic_model_new2")
        print("Training finished. Model saved as ppo_traffic_model_new2.zip")
    elif mode == "eval":
        if not os.path.exists("ppo_traffic_model_new2.zip"):
            print("Trained model not found. Please run in train mode first.")
            sys.exit(0)
        model = PPO.load("ppo_traffic_model_new2")

        episodes = 2

        # Evaluate fixed-time policy over multiple episodes
        fixed_rewards = []
        fixed_queue_records = []
        for i in range(episodes):
            fixed_env = TrafficIntersectionEnv()  # reinitialize env for each episode
            reward, queue_record = evaluate_fixed_time_and_record(fixed_env, green_duration_steps=15)
            fixed_rewards.append(reward)
            fixed_queue_records.append(queue_record)
        avg_fixed_reward = np.mean(fixed_rewards)
        max_fixed_reward = np.max(fixed_rewards)
        fixed_queue_records_avg = np.mean(np.array(fixed_queue_records), axis=0)

        # Evaluate random policy over multiple episodes
        random_rewards = []
        random_queue_records = []
        for i in range(episodes):
            random_env = TrafficIntersectionEnv()
            reward, queue_record = evaluate_random_and_record(random_env)
            random_rewards.append(reward)
            random_queue_records.append(queue_record)
        avg_random_reward = np.mean(random_rewards)
        max_random_reward = np.max(random_rewards)
        random_queue_records_avg = np.mean(np.array(random_queue_records), axis=0)

        # Evaluate PPO policy over multiple episodes
        ppo_rewards = []
        ppo_queue_records = []
        print("Starting the evaluation of PPO... Can take a while lol")
        for i in range(episodes):
            ppo_env = TrafficIntersectionEnv()
            reward, queue_record = evaluate_trained_policy_and_record(ppo_env, model)
            ppo_rewards.append(reward)
            ppo_queue_records.append(queue_record)
        avg_ppo_reward = np.mean(ppo_rewards)
        min_ppo_reward = np.min(ppo_rewards)
        ppo_queue_records_avg = np.mean(np.array(ppo_queue_records), axis=0)

        print(f"Fixed-time policy average total reward: {avg_fixed_reward:.2f}")
        print(f"Fixed-time policy max total reward: {max_fixed_reward:.2f}")
        print(f"Random policy average total reward: {avg_random_reward:.2f}")
        print(f"Random policy max total reward: {max_random_reward:.2f}")
        print(f"PPO policy average total reward: {avg_ppo_reward:.2f}")
        print(f"PPO policy min total reward: {min_ppo_reward:.2f}")

        # Plot the average queue lengths over time for each policy
        plt.figure(figsize=(12, 6))
        plt.plot(fixed_queue_records_avg, label="Fixed-time Policy")
        # plt.plot(random_queue_records_avg, label="Random Policy")
        plt.plot(ppo_queue_records_avg, label="PPO Policy")
        plt.xlabel("Time steps")
        plt.ylabel("Average Total Queue Length (vehicles)")
        plt.title("Average Queue Lengths Over Time (10 Episodes) for Different Policies")
        plt.legend()
        plt.tight_layout()
        plt.show()
