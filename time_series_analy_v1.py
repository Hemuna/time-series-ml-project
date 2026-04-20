import numpy as np
import matplotlib.pyplot as plt

# ============================
# 30 days of step counts
# ============================
steps = np.array([
    5778, 5237, 5001, 7727, 79, 12398, 1702, 5597, 5454, 5105,
    31325, 19569, 21386, 4762, 4604, 4903, 4737, 4630, 81, 7914,
    5038, 5058, 4898, 3881, 90, 96, 3442, 5836, 9872, 5183
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


# ============================Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

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
