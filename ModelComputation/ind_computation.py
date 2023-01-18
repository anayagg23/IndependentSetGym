import networkx as nx
import time
from networkx.algorithms.approximation import clique

n = 20
p = 0.5
n_iter = 10

start_time = time.time()

for i in range(n_iter):
  g = nx.fast_gnp_random_graph(n, p)
  t = len(clique.maximum_independent_set(g))

end_time = time.time()
print(end_time-start_time)
