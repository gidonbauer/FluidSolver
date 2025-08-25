import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from Utility import read_monitor_file

import sys


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <input monitor file>", file=sys.stderr)
    sys.exit(1)

input_file = sys.argv[1]

df = read_monitor_file(input_file)
required_columns = {"cells-per-radius", "radius", "init. error",
                    "cv-mse(curv)", "cv-mrse(curv)", "cv-runtime [us]",
                    "quad-vol-mse(curv)", "quad-vol-mrse(curv)", "quad-vol-runtime [us]",
                    "quad-reg-mse(curv)", "quad-reg-mrse(curv)", "quad-reg-runtime [us]",
                   }
actual_columns   = set(df.columns)
if not required_columns.issubset(actual_columns):
    print(f"Did not find required columns {required_columns}", file=sys.stderr)
    print(f"Actual columns are {actual_columns}", file=sys.stderr)
    missing_columns = required_columns.difference(required_columns.intersection(actual_columns))
    print(f"Missing columns are {missing_columns}")
    sys.exit(1)

# ==================================================================================================
print(f"mean(init. error) = {df['init. error'].mean():.6e}")
print(f"std(init. error)  = {df['init. error'].std():.6e}")
print('\n')

# ==================================================================================================
print("Convolved VOF curvature calculation:")
print(f"  min(mse)  = {df['cv-mse(curv)'].min():.6e}")
print(f"  max(mse)  = {df['cv-mse(curv)'].max():.6e}")
print(f"  mean(mse) = {df['cv-mse(curv)'].mean():.6e}")
print(f"  std(mse)  = {df['cv-mse(curv)'].std():.6e}")
print()
print(f"  min(mrse)  = {df['cv-mrse(curv)'].min():.6e}")
print(f"  max(mrse)  = {df['cv-mrse(curv)'].max():.6e}")
print(f"  mean(mrse) = {df['cv-mrse(curv)'].mean():.6e}")
print(f"  std(mrse)  = {df['cv-mrse(curv)'].std():.6e}")
print()
print(f"  mean(runtime) = {df['cv-runtime [us]'].mean():.6e}µs")
print(f"  std(runtime)  = {df['cv-runtime [us]'].std():.6e}µs")
print(f"------------------------------------------------------------")
print("Quadratic reconstruction (volume matching) curvature calculation:")
print(f"  min(mse)  = {df['quad-vol-mse(curv)'].min():.6e}")
print(f"  max(mse)  = {df['quad-vol-mse(curv)'].max():.6e}")
print(f"  mean(mse) = {df['quad-vol-mse(curv)'].mean():.6e}")
print(f"  std(mse)  = {df['quad-vol-mse(curv)'].std():.6e}")
print()
print(f"  min(mrse)  = {df['quad-vol-mrse(curv)'].min():.6e}")
print(f"  max(mrse)  = {df['quad-vol-mrse(curv)'].max():.6e}")
print(f"  mean(mrse) = {df['quad-vol-mrse(curv)'].mean():.6e}")
print(f"  std(mrse)  = {df['quad-vol-mrse(curv)'].std():.6e}")
print()
print(f"  mean(runtime) = {df['quad-vol-runtime [us]'].mean():.6e}µs")
print(f"  std(runtime)  = {df['quad-vol-runtime [us]'].std():.6e}µs")
print(f"------------------------------------------------------------")
print("Quadratic reconstruction (center regression) curvature calculation:")
print(f"  min(mse)  = {df['quad-reg-mse(curv)'].min():.6e}")
print(f"  max(mse)  = {df['quad-reg-mse(curv)'].max():.6e}")
print(f"  mean(mse) = {df['quad-reg-mse(curv)'].mean():.6e}")
print(f"  std(mse)  = {df['quad-reg-mse(curv)'].std():.6e}")
print()
print(f"  min(mrse)  = {df['quad-reg-mrse(curv)'].min():.6e}")
print(f"  max(mrse)  = {df['quad-reg-mrse(curv)'].max():.6e}")
print(f"  mean(mrse) = {df['quad-reg-mrse(curv)'].mean():.6e}")
print(f"  std(mrse)  = {df['quad-reg-mrse(curv)'].std():.6e}")
print()
print(f"  mean(runtime) = {df['quad-reg-runtime [us]'].mean():.6e}µs")
print(f"  std(runtime)  = {df['quad-reg-runtime [us]'].std():.6e}µs")
print(f"------------------------------------------------------------")
# ==================================================================================================

# ==================================================================================================
df["bin"] = pd.cut(df["cells-per-radius"], bins=50)
cols = ["radius", "cells-per-radius", "cv-mse(curv)", "cv-mrse(curv)", "quad-vol-mse(curv)", "quad-vol-mrse(curv)", "quad-reg-mse(curv)", "quad-reg-mrse(curv)"]
df_binned = df.groupby("bin")[cols].mean().reset_index('bin')
df_binned = df_binned.merge(df.groupby("bin")[cols].std().reset_index('bin'), left_on='bin', right_on='bin', suffixes=('', '_std'))
# ==================================================================================================

# fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

# heights, edges = np.histogram(df['smooth-mse(curv)'], density=True, bins='auto')
# widths  = np.diff(edges)
# centers = edges[:-1] + 0.5 * widths
# ax[0].bar(centers, heights, widths, color="tab:blue", edgecolor="black")

# ax[0].set_xlabel("Mean squared error of curvature")
# ax[0].set_ylabel("Probability density")

# heights, edges = np.histogram(df['smooth-mrse(curv)'], density=True, bins='auto')
# widths  = np.diff(edges)
# centers = edges[:-1] + 0.5 * widths
# ax[1].bar(centers, heights, widths, color="tab:blue", edgecolor="black")
# ax[1].set_xlabel("Mean relative squared error of curvature")
# ax[1].set_ylabel("Probability density")

# plt.show()

# ==================================================================================================
fig, ax = plt.subplots(ncols=3, figsize=(15, 5), layout='tight')

x_axis = "cells-per-radius"

ax[0].semilogy(df[x_axis], df['cv-mse(curv)'], linestyle="", marker=".", color="tab:blue", label="MSE")
ax[0].semilogy(df[x_axis], df['cv-mrse(curv)'], linestyle="", marker=".", color="tab:orange", label="MRSE")
ax[0].legend()
ax[0].set_title("Convolved VOF curvature calculation")

ax[1].semilogy(df[x_axis], df['quad-vol-mse(curv)'], linestyle="", marker=".", color="tab:blue", label="MSE")
ax[1].semilogy(df[x_axis], df['quad-vol-mrse(curv)'], linestyle="", marker=".", color="tab:orange", label="MRSE")
ax[1].legend()
ax[1].set_title("Quadratic volume matching curvature calculation")

ax[2].semilogy(df[x_axis], df['quad-reg-mse(curv)'], linestyle="", marker=".", color="tab:blue", label="MSE")
ax[2].semilogy(df[x_axis], df['quad-reg-mrse(curv)'], linestyle="", marker=".", color="tab:orange", label="MRSE")
ax[2].legend()
ax[2].set_title("Quadratic regression curvature calculation")

for axi in ax:
    if x_axis == "radius":
        axi.set_xlabel("Radius of circle")
    else:
        axi.set_xlabel("Number of cells per radius of circle")
    axi.set_ylabel("Error of curvature")


    y_min = min([df['cv-mse(curv)'].min(),
                 df['cv-mrse(curv)'].min(),
                 df['quad-vol-mse(curv)'].min(),
                 df['quad-vol-mrse(curv)'].min(),
                 df['quad-reg-mse(curv)'].min(),
                 df['quad-reg-mrse(curv)'].min(),
                ])
    y_max = max([df['cv-mse(curv)'].max(),
                 df['cv-mrse(curv)'].max(),
                 df['quad-vol-mse(curv)'].max(),
                 df['quad-vol-mrse(curv)'].max(),
                 df['quad-reg-mse(curv)'].max(),
                 df['quad-reg-mrse(curv)'].max(),
                ])
    axi.set_ylim((y_min/2, y_max*2))

plt.show()
# ==================================================================================================

# ==================================================================================================
fig, ax = plt.subplots(ncols=2, figsize=(10, 5), layout='tight')

ax[0].errorbar(df_binned[x_axis], df_binned["cv-mse(curv)"], df_binned["cv-mse(curv)_std"], linestyle="", marker=".", label="CV")
ax[0].errorbar(df_binned[x_axis], df_binned["quad-vol-mse(curv)"], df_binned["quad-vol-mse(curv)_std"], linestyle="", marker=".", label="Quad. vol.")
ax[0].errorbar(df_binned[x_axis], df_binned["quad-reg-mse(curv)"], df_binned["quad-reg-mse(curv)_std"], linestyle="", marker=".", label="Quad. reg.")
ax[0].set_title("Mean squared error")

ax[1].errorbar(df_binned[x_axis], df_binned["cv-mrse(curv)"], df_binned["cv-mrse(curv)_std"], linestyle="", marker=".", label="CV")
ax[1].errorbar(df_binned[x_axis], df_binned["quad-vol-mrse(curv)"], df_binned["quad-vol-mrse(curv)_std"], linestyle="", marker=".", label="Quad. vol.")
ax[1].errorbar(df_binned[x_axis], df_binned["quad-reg-mrse(curv)"], df_binned["quad-reg-mrse(curv)_std"], linestyle="", marker=".", label="Quad. reg.")
ax[1].set_title("Relative mean squared error")

# ax.semilogy(df_binned[x_axis], df_binned["cv-mrse(curv)"], linestyle="", marker=".", label="CV")
# ax.semilogy(df_binned[x_axis], df_binned["quad-vol-mrse(curv)"], linestyle="", marker=".", label="Quad. vol.")
# ax.semilogy(df_binned[x_axis], df_binned["quad-reg-mrse(curv)"], linestyle="", marker=".", label="Quad. reg.")

for axi in ax:
    if x_axis == "radius":
        axi.set_xlabel("Radius of circle")
    else:
        axi.set_xlabel("Number of cells per radius of circle")
    axi.set_ylabel("Error of curvature")
    axi.set_yscale('log')
    axi.legend()

plt.show()
# ==================================================================================================