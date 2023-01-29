import networkx as nx
import time
from networkx.algorithms.approximation import clique
import numpy as np
import matplotlib.pyplot as plt

n_iter = 10

def compute(n):
  p = 2*np.log(n)/n
  start_time = time.time()
  for i in range(n_iter):
    g = nx.fast_gnp_random_graph(n,p)
    t = len(clique.maximum_independent_set(g))
  end_time = time.time()
  return (end_time-start_time)/10

x = []
y1 = []
y2 = []

for i in range(10, 100):
  x.append(i)
  y1.append(10*compute(i))
  y2.append(np.exp(0.072*i/2.9))

plt.plot(x, y1,'ro')
plt.plot(x, y2,'bo')
plt.xlabel('Number of Vertices')
plt.ylabel('Avg. Computation Time (s)')
plt.show()

