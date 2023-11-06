import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp


# 入力例
n, m = 4, 4
C = np.array([
    [0, 2, 2, 2],
    [2, 0, 1, 2],
    [2, 1, 0, 2],
    [2, 2, 2, 0]]
)
a = np.array([0.2, 0.5, 0.2, 0.1])
b = np.array([0.3, 0.3, 0.4, 0.0])
eps = 0.2

# plot a, b in 1D
plt.figure(figsize=(4, 4))
plt.bar(np.arange(n), a, color="blue", label="a")
plt.xticks(np.arange(n))
plt.title("a")
plt.legend()
plt.show()
plt.bar(np.arange(n), b, color="red", label="b")
plt.xticks(np.arange(n))
plt.title("b")
plt.legend()
plt.show()


# シンクホーンアルゴリズム
K = np.exp(- C / eps)
u = np.ones(n)
for i in range(100):
    v = b / (K.T @ u)
    u = a / (K @ v)
P = u.reshape(n, 1) * K * v.reshape(1, m)
d = (C * P).sum()
print(P)
# 図示
plt.figure(figsize=(4, 4))
plt.imshow(P, cmap="Blues")
plt.colorbar()
plt.title(f"Sinkhorn distance: {d:.3f}")
plt.show()

# plot transport plan
plt.figure(figsize=(4, 4))
for i in range(n):
    for j in range(m):
        plt.plot([i, j], [j, i], color="black", alpha=P[i, j])
plt.title(f"Sinkhorn distance: {d:.3f}")
plt.show()





