import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import matplotlib.animation as animation

n,m = 5, 3

# gradient flow of Sinkhorn divergence
def Sinkhorn(a, b, x, y, eps):
    C = jnp.zeros((n, m))
    for i in range(n):
        for j in range(m):
            C = C.at[i,j].set(jnp.linalg.norm(x[i] - y[j]))
    K = jnp.exp(- C / eps)
    u = jnp.ones(n)
    v = jnp.ones(m)
    for i in range(100):
        v = v.at[:].set(b / (K.T @ u))
        u = u.at[:].set(a / (K @ v))
    P = u.reshape(n, 1) * K * v.reshape(1, m)
    d = (C * P).sum()
    return d, P

x = jax.random.uniform(jax.random.PRNGKey(0), (n, 2))
y = jax.random.uniform(jax.random.PRNGKey(1), (m, 2))
original_x = x.copy()
a = jnp.ones(n) / n
b = jnp.ones(m) / m
eps = 0.01
d = 1e10
i = 0


fig = plt.figure()
plt.scatter(x[:, 0], x[:, 1], color="blue", label="x", alpha=0.5)
plt.scatter(y[:, 0], y[:, 1], color="red", label="y", alpha=0.5)
d, P = Sinkhorn(a, b, x, y, eps)
plt.title(f"Sinkhorn distance: {d:.3f}")
for i in range(n):
    for j in range(m):
        # transform P[i,j] to [0,1]
        plt.plot([x[i, 0], y[j, 0]], [x[i, 1], y[j, 1]], color="black", alpha=P[i, j].item())
plt.legend()
plt.savefig("point_cloud.png")
plt.show()



fig = plt.figure()
plt.scatter(x[:, 0], x[:, 1], color="blue", label="x", alpha=0.5)
plt.scatter(y[:, 0], y[:, 1], color="red", label="y", alpha=0.5)
plt.legend()

reward = 0
def gen():
    global d
    i = 0
    while d > 0.02:
        i += 1
        yield i

def update(frame):
    global x, d, i
    if d < 0.02:
        # stop animation
        plt.close()
        return
    # for each frame, update the data stored on each artist.
    d, P = Sinkhorn(a, b, x, y, eps)
    plt.scatter(x[:, 0], x[:, 1], color="blue", label="x", alpha=0.5)
    plt.title(f"Sinkhorn distance: {d:.3f}, step: {frame}")

    # gradient of d wrt x
    grad_x = jax.grad(lambda x: Sinkhorn(a, b, x, y, eps)[0])(x)
    print(grad_x)
    print(d)
    x -= 0.1 * grad_x
    i += 1


def animate(i):
    update(i)
    return

anim = animation.FuncAnimation(fig, animate, frames=gen)

anim.save("sample.gif", writer="pillow")
plt.show()


