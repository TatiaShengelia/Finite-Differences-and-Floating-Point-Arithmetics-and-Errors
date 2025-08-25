import numpy as np
import matplotlib.pyplot as plt


# define the function and its derivatives
def f(x, y):
    # define f(x, y) = sin(xy) / (xy), with f(0, 0) = 1 to avoid division by zero
    if x == 0 or y == 0:
        return 1.0
    result = np.sin(x * y) / (x * y)
    return result


def f_exact_dx(x, y):
    # exact partial derivative with respect to x
    if x == 0 and y == 0:
        return 0
    return np.cos(x * y) / x - np.sin(x * y) / (y * x ** 2)


def f_exact_dy(x, y):
    # exact partial derivative with respect to y
    if x == 0 and y == 0:
        return 0
    return np.cos(x * y) / y - np.sin(x * y) / (y ** 2 * x)


# finite difference methods
def forward_difference(f, x, y, h, direction='x'):
    if direction == 'x':
        return (f(x + h, y) - f(x, y)) / h
    elif direction == 'y':
        return (f(x, y + h) - f(x, y)) / h


def backward_difference(f, x, y, h, direction='x'):
    if direction == 'x':
        return (f(x, y) - f(x - h, y)) / h
    elif direction == 'y':
        return (f(x, y) - f(x, y - h)) / h


def central_difference(f, x, y, h, direction='x'):
    if direction == 'x':
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    elif direction == 'y':
        return (f(x, y + h) - f(x, y - h)) / (2 * h)


# Richardson's extrapolation using central differences
def richardson_extrapolation(f, x, y, h, direction='x'):
    if direction == 'x':
        return (4 * central_difference(f, x, y, h, direction='x') - central_difference(f, x, y, 2 * h,
                                                                                       direction='x')) / 3
    elif direction == 'y':
        return (4 * central_difference(f, x, y, h, direction='y') - central_difference(f, x, y, 2 * h,
                                                                                       direction='y')) / 3


def tangent_plane(x, y, x0, y0):
    return f(x0, y0) + f_exact_dx(x0, y0) * (x - x0) + f_exact_dy(x0, y0) * (y - y0)


# parameters
x0, y0 = 0.5, 1.0  # the point at which we compute derivatives
h = 0.1

# create grid points around (x0, y0)
x = np.linspace(x0 - 0.5, x0 + 0.5, 100)
y = np.linspace(y0 - 0.5, y0 + 0.5, 100)
X, Y = np.meshgrid(x, y)

# calculate function and finite differences values
Z = np.zeros_like(X)

fd_dx1 = np.zeros_like(X)
bd_dx1 = np.zeros_like(X)
cd_dx1 = np.zeros_like(X)
re_dx1 = np.zeros_like(X)

fd_dy1 = np.zeros_like(Y)
bd_dy1 = np.zeros_like(Y)
cd_dy1 = np.zeros_like(Y)
re_dy1 = np.zeros_like(Y)

for i in range(len(x)):
    for j in range(len(y)):
        Z[i, j] = f(X[i, j], Y[i, j])

        # finite difference approximations (with respect to x)
        fd_dx1[i,j] = forward_difference(f, X[i,j], Y[i,j], h, direction='x')
        bd_dx1[i,j] = backward_difference(f, X[i,j], Y[i,j], h, direction='x')
        cd_dx1[i,j] = central_difference(f, X[i,j], Y[i,j], h, direction='x')
        re_dx1[i,j] = richardson_extrapolation(f, X[i,j], Y[i,j], h, direction='x')

        # finite difference approximations (with respect to y)
        fd_dy1[i,j] = forward_difference(f, X[i,j], Y[i,j], h, direction='y')
        bd_dy1[i,j] = backward_difference(f, X[i,j], Y[i,j], h, direction='y')
        cd_dy1[i,j] = central_difference(f, X[i,j], Y[i,j], h, direction='y')
        re_dy1[i,j] = richardson_extrapolation(f, X[i,j], Y[i,j], h, direction='y')

# calculate tangent plane
Z_tangent = tangent_plane(X, Y, x0, y0)

# calculate errors
error_fd_x1 = abs(fd_dx1 - Z_tangent)
error_bd_x1 = abs(bd_dx1 - Z_tangent)
error_cd_x1 = abs(cd_dx1 - Z_tangent)
error_re_x1 = abs(re_dx1 - Z_tangent)

# calculate errors
error_fd_y1 = abs(fd_dy1 - Z_tangent)
error_bd_y1 = abs(bd_dy1 - Z_tangent)
error_cd_y1 = abs(cd_dy1 - Z_tangent)
error_re_y1 = abs(re_dy1 - Z_tangent)

errors = [[error_fd_x1, error_fd_y1, "Forward Difference"], [error_bd_x1, error_bd_y1, "Backward Difference"],
          [error_cd_x1, error_cd_y1, "Central Difference"], [error_re_x1, error_re_y1, "Richardson Extrapolation"]]


# plot the original function and tangent plane together
plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
surf1 = ax.plot_surface(X, Y, Z, color='red', alpha=0.7)
surf2 = ax.plot_surface(X, Y, Z_tangent, color='blue', alpha=0.7)
ax.set_title('Function (red) vs Tangent Plane (blue)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.show()

for error in errors:
    # plot the errors
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, error[0])
    ax1.set_title(f'error = |{error[2]} - Tangent Plane| with respect to x')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('error with respect to x')

    # Plot 2: Error between numerical derivative and tangent plane
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, error[1])
    ax2.set_title(f'error = |{error[2]} - Tangent Plane| with respect to y')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('error with respect to y')

    plt.tight_layout()
    plt.show()


# comparison of errors for different finite difference formulas
h_values = np.logspace(-1, -10, 30)  # step sizes

errors_fd_x = []  # errors for forward difference (with respect to x)
errors_bd_x = []  # errors for backward difference (with respect to x)
errors_cd_x = []  # errors for central difference (with respect to x)
errors_re_x = []  # errors for Richardson's extrapolation (with respect to x)

errors_fd_y = []  # errors for forward difference (with respect to y)
errors_bd_y = []  # errors for backward difference (with respect to y)
errors_cd_y = []  # errors for central difference (with respect to y)
errors_re_y = []  # errors for Richardson's extrapolation (with respect to y)

# errors for each method
for h in h_values:
    # exact derivatives
    exact_dx = f_exact_dx(x0, y0)
    exact_dy = f_exact_dy(x0, y0)

    # finite difference approximations (with respect to x)
    fd_dx = forward_difference(f, x0, y0, h, direction='x')
    bd_dx = backward_difference(f, x0, y0, h, direction='x')
    cd_dx = central_difference(f, x0, y0, h, direction='x')
    re_dx = richardson_extrapolation(f, x0, y0, h, direction='x')

    # calculate errors
    errors_fd_x.append(abs(fd_dx - exact_dx))
    errors_bd_x.append(abs(bd_dx - exact_dx))
    errors_cd_x.append(abs(cd_dx - exact_dx))
    errors_re_x.append(abs(re_dx - exact_dx))

    # finite difference approximations (with respect to y)
    fd_dy = forward_difference(f, x0, y0, h, direction='y')
    bd_dy = backward_difference(f, x0, y0, h, direction='y')
    cd_dy = central_difference(f, x0, y0, h, direction='y')
    re_dy = richardson_extrapolation(f, x0, y0, h, direction='y')

    # calculate errors
    errors_fd_y.append(abs(fd_dy - exact_dy))
    errors_bd_y.append(abs(bd_dy - exact_dy))
    errors_cd_y.append(abs(cd_dy - exact_dy))
    errors_re_y.append(abs(re_dy - exact_dy))

# plot errors
plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_fd_x, 'o-', label='Forward Difference')
plt.loglog(h_values, errors_bd_x, 's-', label='Backward Difference')
plt.loglog(h_values, errors_cd_x, '^-', label='Central Difference')
plt.loglog(h_values, errors_re_x, 'd-', label="Richardson's Extrapolation")
plt.xlabel('Step size (h)')
plt.ylabel('Absolute Error')
plt.title('Error Analysis for Finite Difference Methods with respect to x')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

plt.figure(figsize=(10, 6))
plt.loglog(h_values, errors_fd_y, 'o-', label='Forward Difference')
plt.loglog(h_values, errors_bd_y, 's-', label='Backward Difference')
plt.loglog(h_values, errors_cd_y, '^-', label='Central Difference')
plt.loglog(h_values, errors_re_y, 'd-', label="Richardson's Extrapolation")
plt.xlabel('Step size (h)')
plt.ylabel('Absolute Error')
plt.title('Error Analysis for Finite Difference Methods with respect to y')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()
