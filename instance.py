import gym
import independent_set_gym
import networkx as nx

env=gym.make("IndependentSet-v0", graph=nx.dodecahedral_graph())
env.reset()
env.step(2)
env.step(3)
env.render()
