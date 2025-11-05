import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simpson
import h5py as h5

import sys

def analytic_u_profile(y, dpdx, dy=0.0):
    mu = 1e-3
    # return dpdx/(2*mu) * (y**2 - y - (dy/2) - (dy/2)**2)
    return dpdx/(2*mu) * (y**2 - y)


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <input HDF5 file>", file=sys.stderr)
    sys.exit(1)

input_file = sys.argv[1]


# = Read data ======================================================================================
file = h5.File(input_file, 'r')

x = file['xcoords'][:]
y = file['ycoords'][:]
xm = x[:-1] + 0.5*np.diff(x)
ym = y[:-1] + 0.5*np.diff(y)
Y, X = np.meshgrid(y, x)

NX = x.shape[0] - 1
NY = y.shape[0] - 1

P = file['60']['pressure'][:, :, 0].reshape((NY, NX)).T
U = file['60']['velocity_x'][:, :, 0].reshape((NY, NX)).T
V = file['60']['velocity_y'][:, :, 0].reshape((NY, NX)).T

# = Plot solution ==================================================================================
fig, ax = plt.subplots(nrows=3, figsize=(15, 6), layout='tight')

c = ax[0].pcolormesh(X, Y, P)
plt.colorbar(c)
ax[0].set_xlabel(R"$x$")
ax[0].set_ylabel(R"$y$")
ax[0].set_title(R"Pressure")

c = ax[1].pcolormesh(X, Y, U)
plt.colorbar(c)
ax[1].set_xlabel(R"$x$")
ax[1].set_ylabel(R"$y$")
ax[1].set_title(R"Velocity $x$")

c = ax[2].pcolormesh(X, Y, V)
plt.colorbar(c)
ax[2].set_xlabel(R"$x$")
ax[2].set_ylabel(R"$y$")
ax[2].set_title(R"Velocity $y$")

plt.show()

# = Plot velocity vs. analytical solution ==========================================================
plt.figure()

idx = 3*NX//4
plt.plot(ym, U[idx, :], label=f"Simulation")

dx = x[1] - x[0]
dy = y[1] - y[0]
dpdx = (P[idx+1, NY//2] - P[idx-1, NY//2]) / (2*dx)
print(f"dpdx = {dpdx}")
plt.plot(ym, analytic_u_profile(ym, dpdx, dy), label=f"Analytical", linestyle="--")

L1_error = simpson(np.abs(analytic_u_profile(ym, dpdx, dy) - U[idx, :]), ym)
plt.text(0.5, 0.1, f"L1 error = {L1_error:.8f}", ha='center', bbox={'boxstyle': 'square', 'fill': False})

plt.xlabel(R"$y$")
plt.ylabel(R"$U(x, y)$")
plt.title(F"$x={xm[idx]:.4f}$")
plt.legend()
plt.xlim((0, 1))
plt.ylim((0, None))

plt.show()

# = Plot pressure profile ==========================================================================
fig, ax = plt.subplots(ncols=2, layout='tight', figsize=(10, 5))

ax[0].plot(xm, P[:, NY//2])
ax[0].set_xlabel(R"$x$")
ax[0].set_ylabel(R"$p(x)$")
ax[0].set_title(R"Pressure")

ax[1].plot(xm[:-1], np.diff(P[:, NY//2])/dx)
ax[1].set_xlabel(R"$x$")
ax[1].set_ylabel(R"$\partial p/\partial x$")
ax[1].set_title(R"Derivative of pressure")

plt.show()
