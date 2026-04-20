import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.signal import periodogram

# 数据（每日步数）
steps = np.array([
    5230, 8120, 7600, 9100, 10200, 12000, 15000,
    6000, 7000, 8500, 9200, 11000, 13000, 16000,
    5500, 7800, 8200, 9700, 10500, 12500, 15500,
    5800, 7300, 8000, 9500, 10800, 12800, 15800,
    6200, 7500
])

t = np.arange(len(steps))

# 1. Time series plot
plt.figure(figsize=(6, 4))
plt.plot(t, steps, marker='o')
plt.title("Time Course of Daily Steps")
plt.xlabel("Day")
plt.ylabel("Steps")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Autocorrelation Function (ACF)
fig, ax = plt.subplots(figsize=(6, 4))
plot_acf(steps, lags=10, ax=ax)
ax.set_title("Autocorrelation Function")
plt.tight_layout()
plt.show()

# 3. Periodogram
f, Pxx = periodogram(steps)

plt.figure(figsize=(6, 4))
plt.plot(f, Pxx)
plt.title("Periodogram")
plt.xlabel("Frequency")
plt.ylabel("Power")
plt.grid(True)
plt.tight_layout()
plt.show()