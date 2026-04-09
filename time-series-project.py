import numpy as np
import matplotlib.pyplot as plt

# ============================
# 30 days of step counts
# ============================
steps = np.array([
    5234, 6890, 10234, 3456, 7890, 12000, 9800, 11000, 7500, 6400,
    8300, 9200, 10500, 4800, 7600, 8900, 10000, 11200, 9700, 6800,
    5400, 7200, 8800, 9300, 10100, 11500, 9800, 7600, 8400, 9000
])

days = np.arange(1, len(steps) + 1)  # Day 1 ~ Day 30


# ============================
# Time course
# ============================
plt.figure(figsize=(10, 4))
plt.plot(days, steps, marker='o')
plt.title("Time Course of Daily Steps (30 Days)")
plt.xlabel("Day")
plt.ylabel("Steps")
plt.xticks(days)  
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================
# Histogram
# ============================
plt.figure(figsize=(10, 4))
plt.hist(steps, bins=10, edgecolor='black')
plt.title("Histogram of Daily Steps")
plt.xlabel("Steps")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ============================
# Scatter plot (y_n vs y_(n-k))
# ============================
k = 1  
plt.figure(figsize=(6, 6))
plt.scatter(steps[:-k], steps[k:], alpha=0.7)
plt.title(f"Scatter Plot (y_n vs y_(n-{k}))")
plt.xlabel("y_n")
plt.ylabel(f"y_(n-{k})")
plt.grid(True)
plt.tight_layout()
plt.show()
