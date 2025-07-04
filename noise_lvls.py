import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0,5,50)
y = x/np.sqrt(0.1)

#plt.plot(x,y)
#plt.show()

n_delta = np.abs(y[4]-y[5])

print(np.max(y))
print(np.min(x))
print(n_delta)