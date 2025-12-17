import numpy as np
from matplotlib import pyplot as plt
from Utility import read_monitor_file
from scipy.integrate import simpson
import sys

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <input monitor file>", file=sys.stderr)
    sys.exit(1)

df = read_monitor_file(sys.argv[1])

# r_expected = 0.1 + 1/2 * 0.05 * df['time'].to_numpy()**2
# r_expected = 0.1 + 1/3 * 0.05 * df['time'].to_numpy()**3

R0 = 0.1
M_DOT = 0.1
RHO_G = 1.0
RHO_L = 1e3
X_MIN = -1.0
X_MAX = 1.0
Y_MIN = -1.0
Y_MAX = 1.0

# = Plot radius ====================================================================================
k = M_DOT / (2*np.pi*RHO_G)
r_expected = np.sqrt(2*k*df['time'].to_numpy() + R0**2)
L1_error = simpson(np.abs(df['r'] - r_expected), df['time']) / simpson(np.abs(r_expected), df['time'])
rel_error = np.abs(r_expected[-1] - df['r'].iloc[-1]) / np.abs(r_expected[-1])
print("")

plt.figure()

plt.plot(df['time'], df['r'], label="Simulation")
plt.plot(df['time'], r_expected, label="Expected", linestyle='--')
plt.xlabel("Time", fontsize=14)
plt.ylabel("Radius", fontsize=14)
plt.legend(fontsize=14)

plt.annotate(f"L1 error = {L1_error:.4e}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=14)
plt.annotate(f"Rel. error (t={df['time'].iloc[-1]:.1f}) = {rel_error:.4e}", xy=(0.05, 0.85), xycoords='axes fraction', fontsize=14)

plt.show()

# = Plot mass ======================================================================================
V_gas_expected = np.pi * r_expected**2
V_liquid_expected = (Y_MAX - Y_MIN) * (X_MAX - X_MIN) - V_gas_expected
m_liquid_expected = V_liquid_expected * RHO_L
L1_error = simpson(np.abs(df['m_liquid'] - m_liquid_expected), df['time']) / simpson(np.abs(m_liquid_expected), df['time'])
rel_error = np.abs(m_liquid_expected[-1] - df['m_liquid'].iloc[-1]) / np.abs(m_liquid_expected[-1])

fig, ax = plt.subplots(ncols=2, layout='tight', figsize=(12, 5))

ax[0].plot(df['time'], df['m_liquid'], label="Simulation")
ax[0].plot(df['time'], m_liquid_expected, label="Expected", linestyle='--')
ax[0].set_xlabel("Time", fontsize=14)
ax[0].set_ylabel("Mass", fontsize=14)
ax[0].set_title("Liquid", fontsize=14)
ax[0].legend(fontsize=14)

ax[0].annotate(f"L1 error = {L1_error:.4e}", xy=(0.05, 0.1), xycoords='axes fraction', fontsize=14)
ax[0].annotate(f"Rel. error (t={df['time'].iloc[-1]:.1f}) = {rel_error:.4e}", xy=(0.05, 0.05), xycoords='axes fraction', fontsize=14)

m_gas_expected = np.pi * R0**2 * RHO_G + M_DOT * df['time'].to_numpy()
L1_error = simpson(np.abs(df['m_gas'] - m_gas_expected), df['time']) / simpson(np.abs(m_gas_expected), df['time'])
rel_error = np.abs(m_gas_expected[-1] - df['m_gas'].iloc[-1]) / np.abs(m_gas_expected[-1])

ax[1].plot(df['time'], df['m_gas'], label="Simulation")
ax[1].plot(df['time'], m_gas_expected, label="Expected", linestyle='--')
ax[1].set_xlabel("Time", fontsize=14)
ax[1].set_ylabel("Mass", fontsize=14)
ax[1].set_title("Gas", fontsize=14)
ax[1].legend(fontsize=14)

ax[1].annotate(f"L1 error = {L1_error:.4e}", xy=(0.05, 0.9), xycoords='axes fraction', fontsize=14)
ax[1].annotate(f"Rel. error (t={df['time'].iloc[-1]:.1f}) = {rel_error:.4e}", xy=(0.05, 0.85), xycoords='axes fraction', fontsize=14)

plt.show()
