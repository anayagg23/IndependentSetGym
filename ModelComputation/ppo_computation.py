import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(20, 50, 100)
y = np.exp((x-2.9)/0.72)

plt.plot(x,y,'ro')
plt.xlabel('Number of Vertices')
plt.ylabel('Avg. Computation Time (s)')
plt.show()


