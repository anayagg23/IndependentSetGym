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
y = []

for i in range(10, 100):
  x.append(i)
  y.append(compute(i))


plt.plot(x,y,'ro')
plt.show()
