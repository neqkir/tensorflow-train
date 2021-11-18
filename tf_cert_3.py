##### AMD MI100 GPU roofline model

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

# 11.5 TFLOPs  
GPU_FLOPS=23.1*10e12 # Flops
# 1.2TB/s mem bandwidth
GPU_BW=1.2*10e12 # bytes
GPU_SLOPE=GPU_FLOPS/GPU_BW

x=np.linspace(1,10e20,1000)

inflexion=GPU_FLOPS/GPU_SLOPE

y=np.piecewise(x, [x<inflexion, x>inflexion], [lambda x:GPU_SLOPE*x, GPU_FLOPS])

plt.plot(x, y, '-r', color='#3B7A57')

plt.title('MI100 Roofline Analysis')
plt.xlabel('Arithmetic Intensity (Flops/Byte)', color='#1C2833')
plt.ylabel('FLOPS', color='#1C2833')

plt.semilogx()
plt.semilogy()

plt.legend(loc='upper left')

X=[]
Y=[]

# batch size of 256
X.append(539762.3279)
Y.append(1.98912E+12) # flops

plt.plot(X, Y, 'go', label='marker only')

plt.grid()
plt.show()










