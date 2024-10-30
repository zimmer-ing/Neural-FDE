import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Define adjustable parameters for font sizes and line width
xlabel_fontsize = 18
ylabel_fontsize = 18
title_fontsize = 20
legend_fontsize = 14
tick_labelsize = 14
line_width = 2

path_extrapol_with_time = Path(Path(__file__).parent, 'FDELowPassFilterExtrapolation_with_time')
path_extrapol_without_time = Path(Path(__file__).parent, 'FDELowPassFilterExtrapolation_without_time')
path_reconstruction_with_time = Path(Path(__file__).parent, 'FDELowPassFilterReconstruction_with_time')
path_reconstruction_without_time = Path(Path(__file__).parent, 'FDELowPassFilterReconstruction_without_time')

algo_list = ['FDE', 'NODE']
dataset_kind = ['train', 'val', 'test']

map_dataset = {'train': 'Training', 'val': 'Validation', 'test': 'Test'}
map_algo = {'FDE': 'NFDE', 'NODE': 'NODE'}
map_task = {'extrapol_with_time': 'E+T',
            'extrapol_without_time': 'E',
            'reconstruction_with_time': 'R+T',
            'reconstruction_without_time': 'R'}

seeds=[0,8,42,98]

# Initialize results dictionary
results = {algo: {dataset: {} for dataset in dataset_kind} for algo in algo_list}

for dataset in dataset_kind:
    df_extrapol_with_time = {algo: pd.read_csv(Path(path_extrapol_with_time, f'{algo}_{dataset}_mean_std.csv'), index_col=0) for algo in algo_list}
    df_extrapol_without_time = {algo: pd.read_csv(Path(path_extrapol_without_time, f'{algo}_{dataset}_mean_std.csv'), index_col=0) for algo in algo_list}
    df_reconstruction_with_time = {algo: pd.read_csv(Path(path_reconstruction_with_time, f'{algo}_{dataset}_mean_std.csv'), index_col=0) for algo in algo_list}
    df_reconstruction_without_time = {algo: pd.read_csv(Path(path_reconstruction_without_time, f'{algo}_{dataset}_mean_std.csv'), index_col=0) for algo in algo_list}

    # First plot with plotly to see the results in detail and to decide which plots to use
    fig = go.Figure()
    for algo in algo_list:
        mode = 'lines+markers' if algo == 'NODE' else 'lines'
        fig.add_trace(go.Scatter(x=df_extrapol_with_time[algo].index, y=df_extrapol_with_time[algo]['mean'], mode=mode, name=f'{algo} {map_task["extrapol_with_time"]}', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_extrapol_without_time[algo].index, y=df_extrapol_without_time[algo]['mean'], mode=mode, name=f'{algo} {map_task["extrapol_without_time"]}', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df_reconstruction_with_time[algo].index, y=df_reconstruction_with_time[algo]['mean'], mode=mode, name=f'{algo} {map_task["reconstruction_with_time"]}', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df_reconstruction_without_time[algo].index, y=df_reconstruction_without_time[algo]['mean'], mode=mode, name=f'{algo} {map_task["reconstruction_without_time"]}', line=dict(color='orange')))

    dataset_name = map_dataset[dataset]
    fig.update_layout(title=f'Mean MSE Loss on {dataset_name} Set', yaxis_type="log")
    # fig.show()

    # Enable LaTeX rendering
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    ## Plotting paper ready using matplotlib and save it as PDF
    End_epoch = 100
    # Use subplots and colors for each task, plot until the end epoch
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Define linestyles
    linestyles = {
        'FDE': '-',
        'NODE': '--'
    }

    tasks = ['extrapol_with_time', 'extrapol_without_time', 'reconstruction_with_time', 'reconstruction_without_time']
    titles = ['Extrapolation with Time', 'Extrapolation without Time', 'Reconstruction with Time', 'Reconstruction without Time']

    for ax, task, title in zip(axs.flatten(), tasks, titles):
        for algo in algo_list:
            df_task = eval(f'df_{task}')
            ax.plot(df_task[algo].index[:End_epoch], df_task[algo]['mean'][:End_epoch],
                    label=f'{map_algo[algo]} {map_task[task]}', linestyle=linestyles[algo], color='black', linewidth=line_width)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=xlabel_fontsize)
        ax.set_ylabel('Loss', fontsize=ylabel_fontsize)
        ax.set_title(title, fontsize=title_fontsize)
        ax.grid(True, which="both", ls="--")
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)

    # Find the global min and max values across all tasks and algorithms
    all_min = min(
        df_extrapol_with_time[algo]['mean'][:End_epoch].min() for algo in algo_list
    )
    all_max = max(
        df_extrapol_with_time[algo]['mean'][:End_epoch].max() for algo in algo_list
    )

    for task in ['extrapol_without_time', 'reconstruction_with_time', 'reconstruction_without_time']:
        all_min = min(all_min, min(
            eval(f'df_{task}')[algo]['mean'][:End_epoch].min() for algo in algo_list
        ))
        all_max = max(all_max, max(
            eval(f'df_{task}')[algo]['mean'][:End_epoch].max() for algo in algo_list
        ))

    # Set the same y-axis limits for all subplots
    for ax in axs.flatten():
        ax.set_ylim(all_min*0.95, all_max)

    # Create a legend for the algorithms
    handles_algo = [plt.Line2D([0], [0], color='black', linestyle=linestyles[algo], linewidth=line_width) for algo in algo_list]
    labels_algo = [map_algo[algo] for algo in algo_list]


    # Place the combined legend below the subplots
    fig.legend(handles_algo, labels_algo, loc='upper center', ncol=2, fontsize=legend_fontsize)

    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.9)
    plt.savefig(Path(Path(__file__).parent, f'Lowpass_{dataset}_loss.pdf'))
    plt.show()
    plt.close()




    # Extracting the values for each algorithm and dataset
    epoch = End_epoch
    for algo in algo_list:
        results[algo][dataset]['extrapol_with_time'] = df_extrapol_with_time[algo].iloc[epoch - 1]
        results[algo][dataset]['extrapol_without_time'] = df_extrapol_without_time[algo].iloc[epoch - 1]
        results[algo][dataset]['reconstruction_with_time'] = df_reconstruction_with_time[algo].iloc[epoch - 1]
        results[algo][dataset]['reconstruction_without_time'] = df_reconstruction_without_time[algo].iloc[epoch - 1]

# Formatting functions
def format_mean_std(mean, std):
    mean = float(mean)
    std = float(std)
    mean_str = f"{mean:.2e}"
    std_str = f"({std:.2e})"
    return f"{mean_str} ± {std_str}"

def format_sci(number):
    if number == 0:
        return "0.0"
    if np.isnan(number):
        return "NaN"
    exponent = np.log10(abs(number))
    exponent = int(np.floor(exponent))
    rounded_exponent = 3 * (exponent // 3)
    base = number / (10 ** rounded_exponent)
    base_formatted = f"{base:.1f}"
    if abs(base - 1000) < 0.001:
        base = 1
        rounded_exponent += 3
        base_formatted = f"{base:.3f}"
    if rounded_exponent == 0:
        return f"{base_formatted}"
    else:
        return f"{base_formatted}e{rounded_exponent}"

# Find the best algorithm for each task
best_algo = {dataset: {key: min(results, key=lambda x: float(results[x][dataset][key]['mean'])) for key in results[algo_list[0]][dataset].keys()} for dataset in dataset_kind}

# Apply formatting function to each row
for algo in algo_list:
    for dataset in dataset_kind:
        for key in results[algo][dataset].keys():
            mean_val = float(results[algo][dataset][key]['mean'])
            std_val = float(results[algo][dataset][key]['std'])
            results[algo][dataset][key]['mean'] = format_sci(mean_val)
            results[algo][dataset][key]['std'] = format_sci(std_val)
            results[algo][dataset][key] = format_mean_std(results[algo][dataset][key]['mean'], results[algo][dataset][key]['std'])

map_col_name = {'extrapol_with_time': r'Extrapolation w.t',
                'extrapol_without_time': r'Extrapolation wo.t',
                'reconstruction_with_time': r'Reconstruction  w.t',
                'reconstruction_without_time': r'Reconstruction wo.t'}

for dataset in dataset_kind:
    df_results = pd.DataFrame({task: {algo: results[algo][dataset][task] for algo in algo_list} for task in results[algo_list[0]][dataset].keys()}, index=algo_list)
    df_results.rename(columns=map_col_name, inplace=True)
    latex_table = df_results.to_latex(escape=False,
                                      header=True,
                                      column_format='l|cccc',
                                      label=f'tab:{dataset}_results',
                                      caption=f'{dataset} Results after {epoch} Epochs',
                                      position='H'
                                      )
    print(latex_table)

    reformatted_results = {dataset: {task: [] for task in tasks} for dataset in dataset_kind}

# Initialize a dictionary to hold the reformatted results for LaTeX table
reformatted_results = {dataset: {task: [] for task in tasks} for dataset in dataset_kind}

# Initialize a dictionary to hold the reformatted results for LaTeX table
reformatted_results = {dataset: {task: [] for task in tasks} for dataset in dataset_kind}

# Initialize a dictionary to hold the reformatted results for LaTeX table
reformatted_results = {dataset: {task: [] for task in tasks} for dataset in dataset_kind}

for dataset in dataset_kind:
    for algo in algo_list:
        for task in tasks:
            reformatted_results[dataset][task].append(results[algo][dataset][task])

# Create a LaTeX table in a single column format
latex_tables = []
for dataset in dataset_kind:
    table = r"\begin{table}[H]" + "\n"
    table += r"    \centering" + "\n"
    table += rf"    \caption{{Results after \textbf{{{epoch}}} Epochs for the \textbf{{{map_dataset[dataset]}}} Dataset}}" + "\n"
    table += rf"    \label{{tab:{dataset}_single_column_results}}" + "\n"
    table += r"    \begin{tabular}{l|cc}" + "\n"
    table += r"        \hline" + "\n"
    table += r"        & \multicolumn{2}{c}{Extrapolation} \\" + "\n"
    table += r"        & with Time & without Time \\" + "\n"
    table += r"        \hline" + "\n"

    for algo in algo_list:
        row = f"        {map_algo[algo]} & "
        row += " & ".join(reformatted_results[dataset][task][algo_list.index(algo)] for task in
                          ['extrapol_with_time', 'extrapol_without_time'])
        row += r" \\" + "\n"
        table += row

    table += r"        \hline" + "\n"
    table += r"        & \multicolumn{2}{c}{Reconstruction} \\" + "\n"
    table += r"        & with Time & without Time \\" + "\n"
    table += r"        \hline" + "\n"

    for algo in algo_list:
        row = f"        {map_algo[algo]} & "
        row += " & ".join(reformatted_results[dataset][task][algo_list.index(algo)] for task in
                          ['reconstruction_with_time', 'reconstruction_without_time'])
        row += r" \\" + "\n"
        table += row

    table += r"        \hline" + "\n"
    table += r"    \end{tabular}" + "\n"
    table += r"\end{table}" + "\n"
    latex_tables.append(table)

# Print the LaTeX tables
for latex_table in latex_tables:
    print(latex_table)



