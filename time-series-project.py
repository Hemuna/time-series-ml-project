import numpy as np
import matplotlib.pyplot as plt

#微信步数
steps = np.array([
    5234, 6890, 10234, 3456, 7890, 12000, 9800
])


# 1. Time course（时间序列图）

plt.figure(figsize=(8, 4))
plt.plot(steps, marker='o')
plt.title("Time Course of Daily Steps")
plt.xlabel("Day")
plt.ylabel("Steps")
plt.grid(True)
plt.show()


# 2. Histogram（直方图）

plt.figure(figsize=(8, 4))
plt.hist(steps, bins=8, edgecolor='black')
plt.title("Histogram of Daily Steps")
plt.xlabel("Steps")
plt.ylabel("Frequency")
plt.show()


# 3. Scatter plot (y_n vs y_{n-k})

k = 1  # 你可以改成 2 或 3
plt.figure(figsize=(6, 6))
plt.scatter(steps[:-k], steps[k:], alpha=0.7)
plt.title(f"Scatter Plot (y_n vs y_(n-{k}))")
plt.xlabel("y_n")
plt.ylabel(f"y_(n-{k})")
plt.grid(True)
plt.show()
