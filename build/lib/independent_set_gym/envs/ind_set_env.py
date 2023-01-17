import gym
import networkx as nx
import matplotlib.pyplot as plt

class ind_set(gym.Env):
    def __init__(self, graph):
        self.graph = graph
        self.max_size = nx.maximal_independent_set(graph)
        self.state = None
        self.action_space = gym.spaces.Discrete(len(self.graph.nodes))
        self.observation_space = gym.spaces.Discrete(len(self.graph.nodes))

    def reset(self):
        self.state = None
        return self.state
    
    def step(self, action):
        node = list(self.graph.nodes)[action]
        neighbors = list(self.graph.neighbors(node))
        if self.state is None:
            self.state = set([node])
        else:
            self.state.add(node)
            for neighbor in neighbors:
                if neighbor in self.state:
                    self.state.remove(neighbor)
        reward = len(self.state)
        done = reward == self.max_size
        return self.state, reward, done, {}
    
    def close(self):
        pass  # you can add any necessary cleanup code here

    def render(self, mode='human'):
        pos = nx.kamada_kawai_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=list(self.state), node_color='r')
        plt.show()

graph = nx.Graph()
graph.add_edges_from([('node1', 'node2'), ('node1', 'node3'), ('node2', 'node3'), ('node2', 'node4'), ('node3', 'node4')])
env = ind_set(graph)
state = env.reset()
node_index = list(graph.nodes).index('node1')

if node_index > 0:
    state, reward, done, _ = env.step(node_index - 1)
else:
    state, reward, done, _ = env.step(node_index)

env.render()
env.close()

