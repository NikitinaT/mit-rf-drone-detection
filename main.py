import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Speed of light (m/s)
c = 3e8

# -----------------------------
# 1. Define Node Positions (meters)
# -----------------------------
nodes = np.array([
    [0, 0],
    [1000, 0],
    [1000, 1000],
    [0, 1000]
])

num_nodes = nodes.shape[0]

# -----------------------------
# 2. Define Drone Position
# -----------------------------
true_pos = np.array([400, 600])

# -----------------------------
# 3. Generate Arrival Times
# -----------------------------
def generate_toa(nodes, pos, sigma_t=5e-9):
    distances = np.linalg.norm(nodes - pos, axis=1)
    toa = distances / c
    noise = np.random.normal(0, sigma_t, size=len(toa))
    return toa + noise

toa = generate_toa(nodes, true_pos)

# -----------------------------
# 4. Compute TDoA (reference = node 0)
# -----------------------------
ref = 0
tdoa = toa - toa[ref]
tdoa = tdoa[1:]  # remove reference
nodes_tdoa = nodes[1:]

# -----------------------------
# 5. TDoA Residual Function
# -----------------------------
def tdoa_residual(x, nodes, tdoa, ref_node):
    d_ref = np.linalg.norm(x - ref_node)
    residuals = []
    for i in range(len(nodes)):
        d_i = np.linalg.norm(x - nodes[i])
        residuals.append((d_i - d_ref) - c * tdoa[i])
    return residuals

# -----------------------------
# 6. Solve Using Least Squares
# -----------------------------
x0 = np.mean(nodes, axis=0)  # initial guess
solution = least_squares(
    tdoa_residual,
    x0,
    args=(nodes_tdoa, tdoa, nodes[ref])
)

estimated_pos = solution.x

# -----------------------------
# 7. Error
# -----------------------------
error = np.linalg.norm(estimated_pos - true_pos)

print("True Position:", true_pos)
print("Estimated Position:", estimated_pos)
print("Localization Error (m):", error)

# -----------------------------
# 8. Plot
# -----------------------------
plt.figure(figsize=(6,6))
plt.scatter(nodes[:,0], nodes[:,1], c='blue', label='RF Nodes')
plt.scatter(true_pos[0], true_pos[1], c='green', label='True Drone')
plt.scatter(estimated_pos[0], estimated_pos[1], c='red', label='Estimated Drone')
plt.legend()
plt.grid()
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("TDoA Drone Localization")
plt.axis("equal")
plt.show()