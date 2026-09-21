import numpy as np
import matplotlib.pyplot as plt

nz, nx, dx = 12 , 47202  , 4.0

v = np.fromfile("shots.bin", dtype=np.float32)
print(v.size, v.min(), v.max())   # ALWAYS check this first
#30351 1500.0 2800.0
v = v.reshape (nx, nz) 

plt.imshow(v, cmap="jet", aspect="auto",
           extent=[0, nx*dx, nz*dx, 0])
plt.colorbar(label="v (m/s)")
plt.xlabel("x (m)"); plt.ylabel("depth (m)")
plt.show()
