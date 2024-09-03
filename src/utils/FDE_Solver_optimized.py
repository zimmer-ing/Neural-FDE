# Copyright (C) 2024 Bernd Zimmering
# This work is licensed under a Creative Commons Attribution 4.0 International License (CC-BY-4.0).
# To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/.
#
# If you find this solver useful in your research, please consider citing:
# @InProceedings{zimmering24,
#     title     = {Optimising Neural Fractional Differential Equations for Performance and Efficiency},
#     author    = {Zimmering, Bernd and Coelho, Cec\'{\i}lia and Niggemann, Oliver},
#     booktitle = {Proceedings of the 1st ECAI Workshop on “Machine Learning Meets Differential Equations: From Theory to Applications”},
#     year      = {2024},
#     pages     = {1-24},
#     volume    = {255},
#     series    = {Proceedings of Machine Learning Research},
#     publisher = {PMLR},
#     url       = {}
# }

import torch
from tqdm import tqdm
import math
import warnings
import inspect
import torch.nn as nn

DEBUG = False


# Helper function to compute the Gamma function
def torchgamma(x):
    return torch.exp(torch.special.gammaln(x))


# Main FDE solver function
def FDEint(f, t, y0, alpha, h=None, dtype=torch.float32, DEBUG=False, return_internals=False):
    """
    Solves a fractional differential equation (FDE) using a predictor-corrector method.

    Parameters:
    - f: function (or an nn.Module) that defines the fractional differential equation (dy/dt = f(t, y)).
         It can be either a function or a neural network.
    - t: time points at which the solution is desired (torch.Tensor).
    - y0: initial condition (torch.Tensor).
    - alpha: fractional order of the differential equation (0 < alpha <= 1).
    - h: step size (optional). If not provided, it is set to the smallest time difference in t.
    - dtype: data type for computations (default: torch.float32).
    - DEBUG: if True, shows a progress bar for the integration steps.
    - return_internals: if True, returns additional internal time and solution values.

    Returns:
    - Approximate solution y at the time points t (and optionally internal time and solution values).
    """

    # Ensure alpha is in the correct range and h is positive
    assert 0 < alpha <= 1, "Alpha must be between 0 and 1"
    assert h is None or h > 0, "Step size must be greater than 0 if provided"

    # Check if the time tensor t needs to be expanded to have a batch dimension
    device = y0.device
    if len(t.shape) == 1:
        t = t.unsqueeze(-1).unsqueeze(0).repeat(y0.shape[0], 1, 1)
    elif len(t.shape) == 2 and t.shape[-1] == 1:
        t = t.unsqueeze(0)

    # Ensure y0 has the right shape for batch operations
    if len(y0.shape) == 1:
        y0 = y0.unsqueeze(0)

    # Check if alpha is a parameter in autograd (for optimization purposes)
    alpha_is_in_autograd = isinstance(alpha, nn.Parameter)

    # Convert all relevant tensors to the specified dtype
    t, y0, alpha = t.to(dtype), y0.to(dtype), alpha.to(dtype)

    # Determine the smallest time step in t and adjust h if necessary
    dt_min = torch.min(torch.diff(t.squeeze()))
    if h is None:
        h = dt_min
    elif dt_min < h and dt_min - h > 1e-6:
        warnings.warn(
            f"The minimum time difference in desired time points ({dt_min}) is smaller than the step size h ({h}). Adjusting h to {dt_min}.",
            UserWarning)
        h = dt_min

    # Tensor initializations for storing intermediate results
    alpha = alpha.squeeze()
    batch_size, dim_y = y0.shape
    N = int(torch.ceil((t.max() - t.min()) / h).item()) + 1  # Number of time steps
    y_internal = torch.zeros((batch_size, N + 1, dim_y), device=device, dtype=dtype)
    t_internal = torch.zeros((batch_size, N + 1), device=device, dtype=dtype)
    fk_mem = torch.zeros_like(y_internal)  # Memory for previous evaluations of f
    y_internal[:, 0, :] = y0  # Set initial condition

    # Precompute coefficients for the predictor and corrector steps
    k = torch.arange(1, N + 1, device=device, dtype=dtype)
    b = ((k ** alpha) - (k - 1) ** alpha).unsqueeze(-1)  # Predictor coefficients
    a = ((k + 1) ** (alpha + 1) - 2 * k ** (alpha + 1) + (k - 1) ** (alpha + 1)).unsqueeze(-1)  # Corrector coefficients
    b = torch.cat([torch.zeros((1, 1), device=device, dtype=dtype), b], dim=0)
    a = torch.cat([torch.zeros((1, 1), device=device, dtype=dtype), a], dim=0)

    # Precompute Gamma functions
    gamma_alpha1 = torchgamma(alpha + 1)
    gamma_alpha2 = torchgamma(alpha + 2)

    # Initial function evaluation f(0, y0)
    f0 = f(t[:, 0, :], y0)
    fk_mem[:, 0, :] = f0.clone() if alpha_is_in_autograd else f0

    y_new = y0
    kn = torch.arange(0, N, device=device, dtype=torch.long)

    # Main loop for time stepping
    for j in tqdm(range(1, N + 1), desc="Time Steps Progress", disable=not DEBUG):
        t_act = (j * h).repeat(batch_size).to(dtype)  # Current time step
        t_internal[:, j] = t_act

        # Compute f at the current step (can be a neural network or a function)
        fkj = f(t_act.unsqueeze(-1), y_new)
        fk_mem[:, j, :] = fkj.clone() if alpha_is_in_autograd else fkj

        # Retrieve previously computed function values
        fk = fk_mem[:, :j, :]

        # Predictor step: Estimate the next value using previous steps
        bjk = b[j - kn[:j]]
        y_p = y0 + (h ** alpha / gamma_alpha1) * torch.sum(bjk * fk, dim=1)

        # Corrector step: Refine the predicted value
        ajk = a[(j - kn[:j])[1:]]
        y_new = y0 + (h ** alpha / gamma_alpha2) * (
                f(t_act.unsqueeze(-1), y_p)
                + ((j - 1) ** (alpha + 1) - (j - 1 - alpha) * j ** alpha) * f0
                + torch.sum(ajk * fk[:, 1:, :], dim=1)
        )

        # Store the new value in the internal solution array
        y_internal[:, j, :] = y_new.clone() if alpha_is_in_autograd else y_new

    # Return the solution, optionally with internal values
    return get_outputs(y_internal, t_internal, t) if not return_internals else (
    get_outputs(y_internal, t_internal, t), t_internal, y_internal)


# Function to interpolate results at desired time points
def get_outputs(y_internal, t_internal, t):
    """
    Interpolates the solution values at the desired time points.

    Parameters:
    - y_internal: solution values at internal time points.
    - t_internal: internal time points corresponding to y_internal.
    - t: desired time points.

    Returns:
    - Interpolated solution values at the desired time points.
    """
    batch_size, num_internal_points = t_internal.shape
    _, num_desired_points, _ = t.shape
    _, _, features = y_internal.shape

    # Find indices of time points just before and after each desired time point
    idx = torch.searchsorted(t_internal, t.squeeze(-1), right=True)
    idx_y = idx.unsqueeze(-1).repeat(1, 1, features)

    # Gather the values just before and after the desired time points
    t0 = torch.gather(t_internal, 1, idx - 1).unsqueeze(-1)
    t1 = torch.gather(t_internal, 1, idx).unsqueeze(-1)
    y0 = torch.gather(y_internal, 1, idx_y - 1)
    y1 = torch.gather(y_internal, 1, idx_y)

    # Perform linear interpolation
    return y0 + (y1 - y0) * (t - t0) / (t1 - t0)


if __name__ == "__main__":
    # Define the fractional differential equation as a simple function (dy/dt = -y)
    def fractional_diff_eq(t, x):
        return -x


    from mittag_leffler import ml as mittag_leffler
    import matplotlib.pyplot as plt
    import time
    import numpy as np


    # Define the data types we want to test the solver with
    data_types = [torch.float32, torch.float64]
    results = {}
    num_steps = 1000  # Number of steps for the time discretization

    # Loop over the defined data types (float32 and float64)
    for dtype in data_types:
        print(f"Running solver with dtype: {dtype}")

        # Define the time points for the simulation
        t = torch.linspace(0., 20., num_steps + 1, dtype=dtype)

        # Real values for comparison using the Mittag-Leffler function
        real_values = [mittag_leffler(-i.item() ** 0.6, 0.6) for i in t]

        # Initial condition for the system (batch size of 1)
        y0 = torch.tensor([1., 1.], dtype=dtype)

        # Start the timer
        start_time = time.time()

        # Prepare the initial condition for batch processing
        batch_size = 1
        y0_batch = y0.repeat(batch_size, 1)

        # Solve the fractional differential equation
        solver_values = FDEint(
            fractional_diff_eq,
            t,
            y0_batch,
            torch.tensor([0.6], dtype=dtype).unsqueeze(0),
            h=torch.tensor(20 / num_steps, dtype=dtype),
            dtype=dtype,
            DEBUG=False
        )

        # End the timer
        end_time = time.time()

        # Print the time taken for the solver to run
        print(f'Time taken by solver: {end_time - start_time:.4f} seconds')

        # Plot the solver's output and the real values for comparison
        plt.plot(t.squeeze().detach().numpy(), solver_values[0, :, 0].detach().numpy(), label=f'Solver (dtype={dtype})')
        plt.plot(t.detach().numpy(), real_values, label='Real values')
        plt.legend()
        plt.show()

        # Compute the error between the solver's output and the real values
        real_values_np = np.array(real_values)
        solver_values_np = solver_values[0].detach().numpy()
        error = real_values_np.flatten() - solver_values_np[:, 0].flatten()

        # Print the total error and plot the error over time
        print(f'Total error: {np.sum(np.abs(error)) / len(error):.6f}')
        plt.plot(t.detach().numpy(), error, label=f'Error (dtype={dtype})')
        plt.legend()
        plt.show()

        # Store the results in a dictionary for later comparison
        results[dtype] = {
            'time': end_time - start_time,
            'total_error': np.sum(np.abs(error)) / len(error)
        }

    # Summarize the results for all data types
    print("Results summary:")
    for dtype, result in results.items():
        print(f"dtype: {dtype}, time taken: {result['time']:.4f} seconds, total error: {result['total_error']:.6f}")