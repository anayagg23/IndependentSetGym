import argparse
import ray
from ray import tune
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from ray.rllib.models.catalog import ModelCatalog
import gym
import independent_set_gym
import networkx as nx
from ray.tune.registry import register_env
from independent_set_gym.envs.ind_set_env import ind_set

def create_env():
    from independent_set_gym.envs.ind_set_env import ind_set as env
    return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=6, type=int)
    parser.add_argument("--training-iteration", default=100, type=int) # change depending on graph size
    parser.add_argument("--ray-num-cpus", default=7, type=int)
    args = parser.parse_args()
    ray.init(num_cpus=args.ray_num_cpus)

    env = create_env()
    tune.register_env("myEnv", lambda: config, env(config))

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    tune.run(
        "contrib/AlphaZero",
        stop={"training_iteration": args.training_iteration},
        max_failures=0,
        config={
            "env": "myEnv",
            "env_config": {
                  "graph": nx.dodecahedral_graph()
             },
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
