import torch
import time
import math
import psutil
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from src.utils.Original_FDE_Solver import solve as original_FDE_solver
from src.utils.FDE_Solver_optimized import FDEint as optimized_FDE_solver
from src.utils.mittag_leffler import ml as mittag_leffler
from memory_profiler import memory_usage
import random
import logging
import Constants as CONST
from pathlib import Path

# Set up logging
logging.basicConfig(filename='solver_performance.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

# Set all seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Function to compute error
def compute_error(real_values, solver_values, steps):
    mse = mean_squared_error(real_values, solver_values)
    return mse / steps

# Define fractional differential equation
def fractionalDiffEq(t, x):
    return -x

# Function to profile the memory usage of a solver
def profile_solver(solver, *args):
    try:
        mem_usage = memory_usage((solver, args))
        return mem_usage
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
        logging.error(f"Memory profiling failed: {e}")
        return [0]  # Return a list with a single zero value

if __name__ == '__main__':
    # Set the seed
    set_seed(42)

    # Time step sizes to test
    dt_values = [1.0, 0.1, 0.01, 0.005]
    num_runs =5   # Number of runs to calculate mean and std

    results = []

    solvers = [
        ('Original Solver', original_FDE_solver),
        ('Optimized Solver', optimized_FDE_solver)
    ]

    name_experiment = Path(__file__).stem
    path = Path(CONST.RESULTS_PATH, name_experiment)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    for dt in dt_values:
        print(f"Testing time step size: {dt}")
        length = 10
        steps = int(length / dt) + 1
        dtype = torch.float64
        t = torch.linspace(0., length, steps, dtype=dtype).unsqueeze(1).cpu()
        real_values = [mittag_leffler(-i.item() ** 0.6, 0.6) for i in t]
        real_values = np.array(real_values)[1:]

        y0 = torch.tensor([1.], dtype=dtype).cpu()

        for run in range(num_runs):
            random.shuffle(solvers)  # Randomize solver order

            # Measure memory usage
            for solver_name, solver in solvers:
                print(f"Run {run + 1}/{num_runs} with solver {solver_name}")
                gc.collect()
                time.sleep(1)  # Sleep for a short time to allow the system to handle other processes

                if solver_name == 'Original Solver':
                    solver_mem = profile_solver(solver, torch.Tensor([0.6]), fractionalDiffEq, y0, t)
                    solver_values = solver(torch.Tensor([0.6]), fractionalDiffEq, y0, t)
                    solver_values = solver_values.squeeze().detach().numpy()
                    # Remove the initial state for error calculation
                    solver_values = solver_values
                else:
                    solver_mem = profile_solver(solver, fractionalDiffEq, t, y0, torch.tensor([0.6], dtype=dtype).unsqueeze(0), torch.tensor(dt), dtype)
                    solver_values = solver(fractionalDiffEq, t, y0, torch.tensor([0.6], dtype=dtype).unsqueeze(0), h=torch.tensor(dt), dtype=dtype)
                    solver_values = solver_values.squeeze().detach().numpy()[1:]

                solver_mem = max(solver_mem)  # Max memory usage during the solver
                results.append((solver_name, dt, None, solver_mem, None))


            # Measure runtime and error
            for solver_name, solver in solvers:
                gc.collect()
                time.sleep(1)  # Sleep for a short time to allow the system to handle other processes

                if solver_name == 'Original Solver':
                    time_start = time.time()
                    solver_values = solver(torch.Tensor([0.6]), fractionalDiffEq, y0, t)
                    time_end = time.time()
                    solver_values = solver_values.squeeze().detach().numpy()
                       # the solver returns too many values
                    solver_values = solver_values[1:]
                else:
                    time_start = time.time()
                    solver_values = solver(fractionalDiffEq, t, y0, torch.tensor([0.6], dtype=dtype).unsqueeze(0), h=torch.tensor(dt), dtype=dtype)
                    time_end = time.time()
                    solver_values = solver_values.squeeze().detach().numpy()[1:]

                solver_time = time_end - time_start

                if len(solver_values) == len(real_values):
                    solver_error = compute_error(real_values, solver_values, steps)
                else:
                    solver_error = float('inf')  # Assign a large error if lengths don't match

                results.append((solver_name, dt, solver_time, None, solver_error))

        # Convert results to DataFrame
        df = pd.DataFrame(results, columns=['Solver', 'dt', 'Time (s)', 'Memory (MB)', 'Error'])

        # Save intermediate results to CSV
        intermediate_path = path / f'solver_performance_intermediate_{dt}.csv'
        df.to_csv(intermediate_path, index=False)

    # Calculate mean and std for each solver and dt
    summary = df.groupby(['Solver', 'dt']).agg(
        Time_mean=('Time (s)', 'mean'),
        Time_std=('Time (s)', 'std'),
        Memory_mean=('Memory (MB)', 'mean'),
        Memory_std=('Memory (MB)', 'std'),
        Error_mean=('Error', 'mean'),
        Error_std=('Error', 'std')
    ).reset_index()

    # Save final summary to CSV
    final_summary_path = path / 'solver_performance_summary.csv'
    summary.to_csv(final_summary_path, index=False)

    # Print the summary
    print(summary)

    # Create LaTeX table (sorted by dt)
    formatted_summary = summary[['Solver', 'dt', 'Time_mean', 'Time_std', 'Memory_mean', 'Memory_std', 'Error_mean', 'Error_std']]
    formatted_summary['Time'] = formatted_summary.apply(lambda row: f"{row['Time_mean']:.2f} ({row['Time_std']:.3f})", axis=1)
    formatted_summary['Memory'] = formatted_summary.apply(lambda row: f"{row['Memory_mean']:.2f} ({row['Memory_std']})", axis=1)
    formatted_summary['Error'] = formatted_summary.apply(lambda row: f"{row['Error_mean']:.2e} ({row['Error_std']:.2e})", axis=1)
    latex_table_1 = formatted_summary.sort_values(by='dt').to_latex(index=False, columns=['Solver', 'dt', 'Time', 'Memory', 'Error'], escape=False)

    # Save the first LaTeX table to file
    latex_path_1 = path / 'solver_performance_summary_by_dt.tex'
    with open(latex_path_1, 'w') as f:
        f.write(latex_table_1)

    # Pivot table to have Solver as columns and dt as rows
    pivot_summary = summary.pivot(index='dt', columns='Solver', values=['Time_mean', 'Time_std', 'Memory_mean', 'Memory_std', 'Error_mean', 'Error_std'])

    # Flatten the column multi-index
    pivot_summary.columns = [f'{metric}_{solver}' for metric, solver in pivot_summary.columns]

    # Create the second LaTeX table
    latex_table_2 = pivot_summary.to_latex(float_format="%.5f", escape=False)

    # Save the second LaTeX table to file
    latex_path_2 = path / 'solver_performance_comparison.tex'
    with open(latex_path_2, 'w') as f:
        f.write(latex_table_2)

    # Print the second LaTeX table to the console
    print("Second LaTeX Table (Solver comparison):")
    print(latex_table_2)

    # Ensure 'Memory_mean' is included in the summary dataframe
    summary = df.groupby(['Solver', 'dt']).agg(
        Time_mean=('Time (s)', 'mean'),
        Time_std=('Time (s)', 'std'),
        Memory_mean=('Memory (MB)', 'mean'),
        Memory_std=('Memory (MB)', 'std'),
        Error_mean=('Error', 'mean'),
        Error_std=('Error', 'std')
    ).reset_index()

    # Pivot table to have Solver as columns and dt as rows
    pivot_summary = summary.pivot(index='dt', columns='Solver', values=['Time_mean', 'Memory_mean', 'Error_mean'])

    # Flatten the column multi-index
    pivot_summary.columns = [f'{metric}_{solver}' for metric, solver in pivot_summary.columns]

    # Create the formatted LaTeX table for the desired output
    latex_table = r"""
    \begin{table}[H]
    \centering
    \begin{tabular}{l|cc|cc|cc}
    \toprule
    dt & \multicolumn{2}{c|}{Time (s)} & \multicolumn{2}{c|}{Memory (MB)} & \multicolumn{2}{c}{Error} \\
    \midrule
     & Original & Optimized & Original & Optimized  & Original & Optimized \\
    \midrule
    """

    # Add the data rows
    for dt in sorted(pivot_summary.index):
        latex_table += f"{dt:.3f} & " \
                       f"{pivot_summary.at[dt, 'Time_mean_Original Solver']:.2f} & " \
                       f"{pivot_summary.at[dt, 'Time_mean_Optimized Solver']:.2f} & " \
                       f"{pivot_summary.at[dt, 'Memory_mean_Original Solver']:.2f} & " \
                       f"{pivot_summary.at[dt, 'Memory_mean_Optimized Solver']:.2f} & " \
                       f"{pivot_summary.at[dt, 'Error_mean_Original Solver']:.2e} & " \
                       f"{pivot_summary.at[dt, 'Error_mean_Optimized Solver']:.2e} \\\\\n"

    # End the table
    latex_table += r"""
    \bottomrule
    \end{tabular}
    \caption{Performance comparison between original and optimized solver}
    \end{table}
    """

    # Save the formatted LaTeX table to file
    latex_path = Path('./solver_performance_comparison_formatted.tex')
    with open(latex_path, 'w') as f:
        f.write(latex_table)

    # Print the formatted LaTeX table to the console
    print("Formatted LaTeX Table:")
    print(latex_table)