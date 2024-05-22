import numpy as np

import matplotlib.pyplot as plt



# Parameters for Lorenz equations

sigma = 10.0

rho = 28.0

beta = 8.0 / 3.0



# Noise strength (Diffusion coefficients)

D_x = 0.1

D_y = 0.1

D_z = 0.1



# Discretization settings

Nx, Ny, Nz = 20, 20, 20  # Number of grid points in x, y, z directions

dx, dy, dz = 2.0, 2.0, 2.0  # Grid spacing

dt = 0.01  # Time step

Nt = 100  # Number of time steps



# Initialize 3D grid for probability density p(x, y, z, t)

p = np.zeros((Nx, Ny, Nz))



# Initial condition: Gaussian distribution centered around (x0, y0, z0)

x0, y0, z0 = 0.0, 0.0, 0.0

for i in range(Nx):

    for j in range(Ny):

        for k in range(Nz):

            x = i * dx

            y = j * dy

            z = k * dz

            p[i, j, k] = np.exp(-((x - x0)**2 + (y - y0)**2 + (z - z0)**2) / (2 * dx))



# Normalize initial distribution

p /= np.sum(p)



# Time evolution using Finite Difference Method

for t in range(Nt):

    # Create a copy to store the updated values

    p_new = np.copy(p)

    

    for i in range(1, Nx - 1):

        for j in range(1, Ny - 1):

            for k in range(1, Nz - 1):

                x = i * dx

                y = j * dy

                z = k * dz

                

                # Drift terms (from stochastic Lorenz equations)

                A_x = sigma * (y - x)

                A_y = x * (rho - z) - y

                A_z = x * y - beta * z

                

                # Finite Difference update for p(x, y, z, t)

                dpdt = (

                    - (A_x * (p[i+1, j, k] - p[i-1, j, k]) / (2 * dx))

                    - (A_y * (p[i, j+1, k] - p[i, j-1, k]) / (2 * dy))

                    - (A_z * (p[i, j, k+1] - p[i, j, k-1]) / (2 * dz))

                    + (D_x * (p[i+1, j, k] - 2 * p[i, j, k] + p[i-1, j, k]) / dx**2)

                    + (D_y * (p[i, j+1, k] - 2 * p[i, j, k] + p[i, j-1, k]) / dy**2)

                    + (D_z * (p[i, j, k+1] - 2 * p[i, j, k] + p[i, j, k-1]) / dz**2)

                )

                

                # Update p(x, y, z, t) using Euler scheme

                p_new[i, j, k] = p[i, j, k] + dt * dpdt

    

    # Update p for the next iteration

    p = np.copy(p_new)



# Final probability density (after Nt time steps)

# For demonstration, we'll plot a slice at z = z0

z_index = int(z0 / dz)

plt.imshow(p[:, :, z_index], extent=[0, Nx*dx, 0, Ny*dy], origin='lower',

           cmap='viridis', aspect='auto')

plt.colorbar(label='Probability Density')

plt.title(f"Probability Density at z = {z0} (after {Nt} time steps)")

plt.xlabel('x')

plt.ylabel('y')

plt.show()
