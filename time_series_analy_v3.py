import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Settings
# -----------------------------
np.random.seed(0)
N = 100
true_lambda = 2
reps = 1000

# -----------------------------
# Figure 1: Log-likelihood plot
# -----------------------------
# Generate one dataset
data = np.random.exponential(scale=1/true_lambda, size=N)

# Log-likelihood function
def log_likelihood(lam, x):
    return len(x) * np.log(lam) - lam * np.sum(x)

lam_grid = np.linspace(0.1, 5, 300)
ll_values = [log_likelihood(lam, data) for lam in lam_grid]

# MLE
lambda_hat = len(data) / np.sum(data)

# Plot
plt.figure(figsize=(6,4))
plt.plot(lam_grid, ll_values, label="log-likelihood")
plt.axvline(lambda_hat, color='red', linestyle='--', label=f"MLE = {lambda_hat:.2f}")
plt.xlabel("λ")
plt.ylabel("log-likelihood")
plt.title("Figure 1. Log-likelihood Function for Exp(λ)")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Figure 2: Histogram of MLE
# -----------------------------
lambda_hats = []

for _ in range(reps):
    sample = np.random.exponential(scale=1/true_lambda, size=N)
    lam_hat = len(sample) / np.sum(sample)
    lambda_hats.append(lam_hat)

plt.figure(figsize=(6,4))
plt.hist(lambda_hats, bins=30, density=True, edgecolor='black', alpha=0.7)
plt.axvline(true_lambda, color='red', linestyle='--', label="True λ = 2")
plt.xlabel("λ̂")
plt.ylabel("Density")
plt.title("Figure 2. Histogram of MLE λ̂")
plt.legend()
plt.tight_layout()
plt.show()
