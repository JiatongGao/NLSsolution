from scipy.io import savemat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Divide the range -5 to just before 5 into 256 points
x = np.linspace(-1, 1, 65)[:-1]
y = np.linspace(-1,1,65)[:-1]
nx = x.shape[0]
ny = y.shape[0]


# z1 = 2 / np.cosh(x) + 2 / np.cosh(y)
# z1 = z1.reshape(-1, 1)
#
t = np.linspace(0, np.pi / 2, 64)


z = np.zeros((nx,ny), dtype=complex)
for i in range(nx):
    for j in range(ny):
        #z[i,j] = np.exp(-x[i] ** 2 - y[j] ** 2)
        #z[i, j] = 2 / (np.cosh(x[i] ** 2 + y[j] ** 2))
        #z[i, j] = np.sin(x[i] ** 2 + y[j] ** 2)
        z[i, j] = np.exp(2 * np.pi * 1j * (1 * (x[i] + y[j])))
        # rho = 1
        # m = 1
        # lam = 1
        # time = t[-1]
        # omega = np.abs(m) ** 2 + lam * (rho ** 2)
        #z[i, j] = rho * np.exp(1j * (m * (x[i]+ y[j]) - omega * time))

zreal = np.real(z)
zimag = np.imag(z)


plt.figure(figsize=(6, 4))
plt.contourf(x, y, zreal.T, cmap='viridis')  # Use contourf for filled contour plot
plt.colorbar(label='Value of z')
#plt.title(f'Simplified Contour Plot of z with Color Bar(Real){time}')
plt.xlabel('x')
plt.ylabel('y')

plt.figure(figsize=(6, 4))
plt.contourf(x, y, zimag.T, cmap='viridis')  # Use contourf for filled contour plot
plt.colorbar(label='Value of z')
#plt.title(f'Simplified Contour Plot of z with Color Bar(Imag){time}')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


data2 = {
    'x': x,
    'y': y,
    'uu': z,
    't': t,
}

mat_file_path = "/Users/jiatonggao/Desktop/datanlstorus_planewave_init_646464.mat"
savemat(mat_file_path, data2)

