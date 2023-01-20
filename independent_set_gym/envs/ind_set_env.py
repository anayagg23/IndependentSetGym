import gym
import networkx as nx
from networkx.algorithms.approximation import clique
import numpy as np
import matplotlib.pyplot as plt
import argparse
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
import timeit
import time

# 1. modify independent variables: reward system and algorithm (PPO -> X)
# 2. work for random graphs CHECK
# 3. curve fit results for mathematical comparison, not just visual with tensorboard

n = 50
r_graph = nx.fast_gnp_random_graph(n, 0.5)
n_iter = 100
p = 2*np.log(n)/n

class ind_set(gym.Env):
    def __init__(self, graph):
        self._max_episode_steps=100
        self.graph = nx.fast_gnp_random_graph(n,p)
        self.max_size = len(clique.maximum_independent_set(self.graph))
        self.state = []
        for i in range(len(self.graph.nodes)):
            self.state.append(0)
        self.action_space = gym.spaces.Discrete(len(self.graph.nodes))
        self.observation_space = gym.spaces.MultiBinary(len(self.graph.nodes))
        self.num_steps = 0

    def reset(self):
        self.num_steps = 0
        self.state = []
        for i in range(len(self.graph.nodes)):
            self.state.append(0)
        return self.state
    
    def step(self, action):
        done = False
        self.num_steps+=1
        node = list(self.graph.nodes)[action]
        neighbors = list(self.graph.neighbors(node))
        self.state[int(node)] = 1
        for neighbor in neighbors:
            if self.state[int(neighbor)] == 1:
                self.state[int(neighbor)] = 0
                reward = np.NINF
        if (sum(x for x in self.state) >= self.max_size):
            reward = 1/(self.num_steps-self.max_size+0.0001)
            done = True
        else:
            reward = 0
            done = False
        return self.state, reward, done, {}
    
    def close(self):
        pass  # you can add any necessary cleanup code here

    def render(self, mode='human'):
        node_colors = ['red' if self.state[int(node)]==1 else 'blue' for node in self.graph.nodes]
        fig, ax = plt.subplots()
        nx.draw_networkx(self.graph, with_labels=True, node_color=node_colors, node_size=1000, node_shape='o', edge_color='black', linewidths=2)
        plt.show()

    def evaluate(self, algo):
        obs = self.reset()
        action = algo.compute_single_action(obs)
        obs, reward, done, info = self.step(action)
        while not done:
            action = algo.compute_single_action(obs)
            obs, reward, done, info = self.step(action)
        print("Resultant Redundancy:", 1/reward, " Resultant Reward:", reward)


config = (
      PPOConfig()
      .environment(
          env = ind_set,
          env_config = {
                "graph": r_graph
          },
      )
      .rollouts(num_rollout_workers=3)
)

algo = config.build()

cumulative_time = 0
num_iterations = n_iter

for i in range(num_iterations):
    start_time = timeit.default_timer()
    results = algo.train()
    end_time = timeit.default_timer()
    total_time = end_time - start_time
    cumulative_time += total_time
    # redundancy = results['episode_len_mean']-np.floor(results['episode_len_mean'])
    print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}; avg. depth={results['episode_len_mean']}; avg. redundancy={1/results['episode_reward_mean']}; total time = {total_time}")

print(f"Cumulative Time = {cumulative_time}")


#onto the testing phase

for i in range(10):
  game = ind_set(r_graph)
  game.reset()

  done = False
  obs = game.state

  start_time = time.time()

  while not done:
    action = algo.compute_single_action(obs)
    obs, reward, done, info = game.step(action)

  end_time = time.time()

  total_time = end_time-start_time

  print("Total time to beat the game: {:.2f} seconds".format(total_time))




'''
def env_creator(env_config):
    return ind_set(env_config["graph"])

tune.register_env("myenv", env_creator)

G = nx.dodecahedral_graph()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=6, type=int)
    parser.add_argument("--training-iteration", default=100, type=int) # change depending on graph size
    parser.add_argument("--ray-num-cpus", default=7, type=int)
    args = parser.parse_args()
    ray.init(num_cpus=args.ray_num_cpus)

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    tune.run(
        "contrib/AlphaZero",
        stop={"training_iteration": args.training_iteration},
        max_failures=0,
        config={
            "env": "myenv",
            "env_config": {"graph": G},
            "num_workers": args.num_workers,
            "rollout_fragment_length": 10,
            "train_batch_size": 50,
            "sgd_minibatch_size": 8,
            "lr": 1e-4,
            "num_sgd_iter": 1,
            "mcts_config": {
                "puct_coefficient": 1.5,
                "num_simulations": 5,
                "temperature": 1.0,
                "dirichlet_epsilon": 0.20,
                "dirichlet_noise": 0.03,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": True,
            },
            "ranked_rewards": {
                "enable": True,
            },
            "model": {
                "custom_model": "dense_model",
            },
        },
    )
'''
