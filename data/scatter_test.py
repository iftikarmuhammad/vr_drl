import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

inp = sio.loadmat('scatter_data.mat', squeeze_me=True)

x = inp['scatter_Head_speed']
y = inp['scatter_Frame_size']

print(x)
print(y)
z = poly.polyfit(x, y, 1)
z2 = poly.polyfit(x, y, 2)
p = poly.Polynomial(z)
p2 = poly.Polynomial(z2)

print(p2)
# ffit = poly.Polynomial(z) 

# print(x)

x = np.array([i for i in range(60)])
plt.plot(x, p(x), color='r', label='Linear regression')
plt.plot(x, p2(x), color='b', label='Quadratic regression')
plt.scatter(inp['scatter_Head_speed'], y, label='Data sample')
plt.axis((0,60,800,2200))
plt.grid(True)
plt.xlabel('Head speed (deg/s)')
plt.ylabel('Frame size (kbytes)')
plt.legend(loc='lower right')
plt.show()