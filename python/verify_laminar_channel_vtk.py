import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson

from vtk import vtkStructuredGridReader
from vtk.util import numpy_support as vtk_np

import sys

def analytic_u_profile(y, dpdx, dy=0.0):
    mu = 1e-3
    # return dpdx/(2*mu) * (y**2 - y - (dy/2) - (dy/2)**2)
    return dpdx/(2*mu) * (y**2 - y)


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <input VTK file>", file=sys.stderr)
    sys.exit(1)

input_file = sys.argv[1]

# = Initialize VTK reader ==========================================================================
reader = vtkStructuredGridReader()
reader.SetFileName(input_file)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()
if reader.GetErrorCode() != 0:
    print(f"Could not read file `{input_file}`", file=sys.stderr)
    sys.exit(1)

# = Read dimensions ================================================================================
data = reader.GetOutput()

point_dims = [-1, -1, -1]
data.GetDimensions(point_dims)
cell_dims = [i-1 for i in point_dims[:2]]

point_dims[0], point_dims[1] = point_dims[1], point_dims[0]
cell_dims[0], cell_dims[1] = cell_dims[1], cell_dims[0]

print(f"{point_dims = }")
print(f"{cell_dims = }")

# = Read grid ======================================================================================
x = np.zeros(data.GetNumberOfPoints())
y = np.zeros(data.GetNumberOfPoints())
z = np.zeros(data.GetNumberOfPoints())

for i in range(data.GetNumberOfPoints()):
        x[i], y[i], z[i] = data.GetPoint(i)

x = x.reshape(point_dims[:2])
y = y.reshape(point_dims[:2])
z = z.reshape(point_dims[:2])

X = np.ndarray(shape=cell_dims)
Y = np.ndarray(shape=cell_dims)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i, j] = 1/4 * (x[i, j] + x[i, j+1] + x[i+1, j] + x[i+1, j+1])
        Y[i, j] = 1/4 * (y[i, j] + y[i, j+1] + y[i+1, j] + y[i+1, j+1])

# Read cells
P = vtk_np.vtk_to_numpy(data.GetCellData().GetArray('pressure')).reshape(cell_dims)
U = vtk_np.vtk_to_numpy(data.GetCellData().GetArray('velocity')).reshape(cell_dims + [3])

# = Plot solution ==================================================================================
fig, ax = plt.subplots(nrows=3, figsize=(15, 6), layout='tight')

c = ax[0].pcolormesh(X, Y, P)
plt.colorbar(c)
ax[0].set_xlabel(R"$x$")
ax[0].set_ylabel(R"$y$")
ax[0].set_title(R"Pressure")

c = ax[1].pcolormesh(X, Y, U[:, :, 0])
plt.colorbar(c)
ax[1].set_xlabel(R"$x$")
ax[1].set_ylabel(R"$y$")
ax[1].set_title(R"Velocity $x$")

c = ax[2].pcolormesh(X, Y, U[:, :, 1])
plt.colorbar(c)
ax[2].set_xlabel(R"$x$")
ax[2].set_ylabel(R"$y$")
ax[2].set_title(R"Velocity $y$")

plt.show()

# # = Plot velocity profile ==========================================================================
# plt.figure()
# for idx in [1, X.shape[1] // 4, 2 * X.shape[1] // 4, 3 * X.shape[1] // 4, X.shape[1] - 2]:
#     plt.plot(Y[:, idx], U[:, idx, 0], label=f"x={X[0, idx]:.4f}")
# plt.xlabel(R"$y$")
# plt.ylabel(F"$U(x, y)$")
# plt.legend()
# plt.show()

# = Plot velocity vs. analytical solution ==========================================================
plt.figure()

idx = 3*X.shape[1]//4 # X.shape[1] - 2
plt.plot(Y[:, idx], U[:, idx, 0], label=f"Simulation")

dx = X[0, 1] - X[0, 0]
dy = Y[1, 0] - Y[0, 0]
dpdx = (P[X.shape[0]//2, idx+1] - P[X.shape[0]//2, idx-1]) / (2*dx)
print(f"dpdx = {dpdx}")
plt.plot(Y[:, idx], analytic_u_profile(Y[:, idx], dpdx, dy), label=f"Analytical", linestyle="--")

L1_error = simpson(np.abs(analytic_u_profile(Y[:, idx], dpdx, dy) - U[:, idx, 0]), Y[:, idx])
plt.text(0.5, 0.1, f"L1 error = {L1_error:.8f}", ha='center', bbox={'boxstyle': 'square', 'fill': False})

plt.xlabel(R"$y$")
plt.ylabel(R"$U(x, y)$")
plt.title(F"$x={X[0, idx]:.4f}$")
plt.legend()
plt.xlim((0, 1))
plt.ylim((0, None))

plt.show()

# = Plot pressure profile ==========================================================================
fig, ax = plt.subplots(ncols=2, layout='tight', figsize=(10, 5))

ax[0].plot(X[X.shape[0]//2, :], P[X.shape[0]//2, :])
ax[0].set_xlabel(R"$x$")
ax[0].set_ylabel(R"$p(x)$")
ax[0].set_title(R"Pressure")

ax[1].plot(X[X.shape[0]//2, :-1], np.diff(P[X.shape[0]//2, :])/dx)
ax[1].set_xlabel(R"$x$")
ax[1].set_ylabel(R"$\partial p/\partial x$")
ax[1].set_title(R"Derivative of pressure")

plt.show()
