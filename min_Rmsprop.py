import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivative
f  = lambda x: 0.1*x**3 - 0.8*x**2 - 1.5*x + 5.4
fd = lambda x: 0.3*x**2 - 1.6*x - 1.5

# Parameters
viz_range = np.array([-6, 12])
learn_rate = 0.01  # Learning rate
max_iter = 100
min_tol = 1e-6
x_init = 12       # Initial point

# RMSprop parameters
beta = 0.9       # Decay rate
epsilon = 1e-8   # Small constant to avoid division by zero
cache = 0        # Initialize cache

# Prepare visualization
xs = np.linspace(*viz_range, 100)
plt.plot(xs, f(xs), 'r-', label='f(x)', linewidth=2)
plt.plot(x_init, f(x_init), 'b.', label='Each step', markersize=12)
plt.axis((*viz_range, *f(viz_range)))
plt.legend()

x = x_init
for i in range(max_iter):
    # Run the RMSprop gradient descent
    xp = x
    grad = fd(x)
    cache = beta * cache + (1 - beta) * grad**2
    x = x - learn_rate * grad / (np.sqrt(cache) + epsilon)

    # Update visualization for each iteration
    print(f'Iter: {i}, x = {xp:.3f} to {x:.3f}, f(x) = {f(xp):.3f} to {f(x):.3f} (f\'(x) = {grad:.3f})')
    lcolor = np.random.rand(3)
    approx = fd(xp) * (xs - xp) + f(xp)
    plt.plot(xs, approx, '-', linewidth=1, color=lcolor, alpha=0.5)
    plt.plot(x, f(x), '.', color=lcolor, markersize=12)

    # Check the terminal condition
    if abs(x - xp) < min_tol:
        break

plt.show()
