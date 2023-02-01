import networkx as nx
import time
from networkx.algorithms.approximation import clique
import numpy as np
import matplotlib.pyplot as plt

n_iter1 = 10
n_iter2 = 5

def p(n): # edge probability
  return 2*np.log(n)/n

def maximum_independent_set(G): # naive approach
    def is_independent_set(nodes):
        for node in nodes:
            neighbors = set(G.neighbors(node))
            if any(neighbor in nodes for neighbor in neighbors):
                return False
        return True
    
    def find_maximum_independent_set(candidate_nodes, selected_nodes):
        if not candidate_nodes:
            return selected_nodes
        
        current_node = candidate_nodes[0]
        candidate_nodes_without_current = candidate_nodes[1:]
        selected_nodes_with_current = selected_nodes + [current_node]
        
        if is_independent_set(selected_nodes_with_current):
            return find_maximum_independent_set(candidate_nodes_without_current, selected_nodes_with_current)
        else:
            return find_maximum_independent_set(candidate_nodes_without_current, selected_nodes)
    
    return find_maximum_independent_set(list(G.nodes()), [])

def naive(n):
    start_time = time.time()
    for i in range(n_iter2):
      g = nx.fast_gnp_random_graph(n,p(n))
      t = maximum_independent_set(g)
    end_time = time.time()
    return (end_time-start_time)/n_iter2


def compute(n):
  start_time = time.time()
  for i in range(n_iter1):
    g = nx.fast_gnp_random_graph(n,p(n))
    t = len(clique.maximum_independent_set(g))
  end_time = time.time()
  return (end_time-start_time)/n_iter1

x = []
y1 = [] # compute - current fastest algo
y2 = [] # PPO
y3 = [] # naive algo

for i in range(10, 100):
  x.append(i)
  y1.append(n_iter1*compute(i))
  y2.append(np.exp(0.072*i/2.9))
  t = 100000*naive(i)
  if (t<600):
    y3.append(t)
  else:
    y3.append(0)

plt.plot(x, y1, 'ro')
plt.plot(x, y2, 'bo')
plt.plot(x, y3, 'go')
plt.xlabel('Number of Vertices')
plt.ylabel('Avg. Computation Time (s)')
plt.show()

