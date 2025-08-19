import numpy as np
from matplotlib import pyplot as plt
import sys

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <input directory> <fields to plot...>", file=sys.stderr)
    sys.exit(1)

input_dir = sys.argv[1]
fields_to_plot = set(map(str.lower, sys.argv[2:]))

x  = np.load(f"{input_dir}/x.npy")
xm = np.load(f"{input_dir}/xm.npy")
y  = np.load(f"{input_dir}/y.npy")
ym = np.load(f"{input_dir}/ym.npy")

visc = np.load(f"{input_dir}/visc.npy")

p = np.load(f"{input_dir}/p.npy")
p_jump_u_stag = np.load(f"{input_dir}/p_jump_u_stag.npy")
p_jump_v_stag = np.load(f"{input_dir}/p_jump_v_stag.npy")

rho_u_stag_old = np.load(f"{input_dir}/rho_u_stag_old.npy")
rho_v_stag_old = np.load(f"{input_dir}/rho_v_stag_old.npy")
U_old = np.load(f"{input_dir}/U_old.npy")
V_old = np.load(f"{input_dir}/V_old.npy")

rho_u_stag = np.load(f"{input_dir}/rho_u_stag.npy")
rho_v_stag = np.load(f"{input_dir}/rho_v_stag.npy")
U = np.load(f"{input_dir}/U.npy")
V = np.load(f"{input_dir}/V.npy")

Y,  X  = np.meshgrid(ym, xm)
YU, XU = np.meshgrid(ym, x)
YV, XV = np.meshgrid(y, xm)

def contains_any(a: set, b: set) -> bool:
    return len(a.intersection(b)) > 0

if contains_any(fields_to_plot, {'u', 'v', 'velocity', 'all'}):
    fig, ax = plt.subplots(nrows=2, figsize=(10, 5), layout='tight')
    
    c = ax[0].pcolormesh(XU, YU, U)
    plt.colorbar(c)
    ax[0].set_title("$U$")

    c = ax[1].pcolormesh(XV, YV, V)
    plt.colorbar(c)
    ax[1].set_title("$V$")
    
    for axi in ax:
        axi.set_xlabel("$x$")
        axi.set_ylabel("$y$")
    
    plt.show()

if contains_any(fields_to_plot, {'rho', 'density', 'all'}):
    fig, ax = plt.subplots(nrows=2, figsize=(10, 5), layout='tight')
    
    c = ax[0].pcolormesh(XU, YU, rho_u_stag)
    plt.colorbar(c)
    ax[0].set_title(R"$\rho^U$")

    c = ax[1].pcolormesh(XV, YV, rho_v_stag)
    plt.colorbar(c)
    ax[1].set_title(R"$\rho^V$")
    
    for axi in ax:
        axi.set_xlabel("$x$")
        axi.set_ylabel("$y$")
    
    plt.show()

if contains_any(fields_to_plot, {'p', 'p_jump', 'pressure', 'all'}):
    fig, ax = plt.subplots(nrows=3, figsize=(15, 5), layout='tight')
    
    c = ax[0].pcolormesh(X, Y, p)
    plt.colorbar(c)
    ax[0].set_title(R"$p$")

    c = ax[1].pcolormesh(XU, YU, p_jump_u_stag)
    plt.colorbar(c)
    ax[1].set_title(R"$p_{\mathrm{jump}}^U$")
    
    c = ax[2].pcolormesh(XV, YV, p_jump_v_stag)
    plt.colorbar(c)
    ax[2].set_title(R"$p_{\mathrm{jump}}^V$")
    
    for axi in ax:
        axi.set_xlabel("$x$")
        axi.set_ylabel("$y$")
    
    plt.show()

if contains_any(fields_to_plot, {'visc', 'all'}):
    fig, ax = plt.subplots(nrows=1, figsize=(5, 5), layout='tight')
    
    c = ax.pcolormesh(X, Y, visc)
    plt.colorbar(c)
    ax.set_title(R"$\nu$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    
    plt.show()