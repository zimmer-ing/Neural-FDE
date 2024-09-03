import hashlib
import numpy as np
import json
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
import pandas as pd
from src.datasets.base import DatasetBase
from pathlib import Path
import Constants as const
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import brainpy as bp
pio.renderers.default = "browser"
from functools import partial

class HomogeneousFDEDataset(DatasetBase):

    def __init__(self, directory='train', file_extension=".csv.gz", data_parameters=None, device=None,config=None):

        if config is not None:
            data_parameters=config['params']

        if data_parameters is None:
            #print warning
            print('No parameter config was provided for the dataset. Using default parameters')
            #use default config
            config = {
                "n_samples": 20,
                "System": 'LorenzSystem',
                "Task": "Reconstruction",
                "TaskParameters": {
                    "initial_values_range_validation": [-1, 1],
                    "initial_values_range_test": [-1, 1],
                },
                "SystemParameters": {
                    "a": 10,
                    "b": 28,
                    "c": 8 / 3
                },
                "alpha": 0.9,
                "noise": 0.0,
                "inital_values_range": [-1, 1],
                "time_step_simulation": 0.01,
                "final_time": 3,
                "seed": 42
            }
        assert data_parameters["noise"] >= 0
        assert data_parameters["inital_values_range"][0] <= data_parameters["inital_values_range"][1]
        assert data_parameters["time_step_simulation"] > 0
        assert data_parameters["final_time"] > 0
        assert data_parameters["seed"] is not None
        assert data_parameters["n_samples"] > 0
        assert data_parameters["System"] in ['LorenzSystem','DampedOscillator','LowPassFilter']
        assert data_parameters["Task"] in ['Reconstruction','Extrapolation','Interpolation']
        assert data_parameters["TaskParameters"] is not None

        self.data_parameters = data_parameters
        hash=generate_hash_from_dict(data_parameters)
        full_directory=Path(const.DATA_PATH,'HomogeneousFDE',str(data_parameters["System"]),str(data_parameters['Task']),hash,directory)
        #set seed for reproducibility

        #check if the dataset is already generated
        if not Path(full_directory,'sample_0'+file_extension).exists():
            full_directory.parent.mkdir(parents=True, exist_ok=True)
            Path(full_directory.parent,'train').mkdir(parents=True, exist_ok=True)
            Path(full_directory.parent,'val').mkdir(parents=True, exist_ok=True)
            Path(full_directory.parent,'test').mkdir(parents=True, exist_ok=True)
            #training data
            #save config
            with open(Path(full_directory.parent,'train','config.json'),'w') as f:
                json.dump(data_parameters, f)
            generate_dataset(data_parameters, Path(full_directory.parent, 'train'))
            #validation data
            data_parameters=mod_task_parameters(data_parameters.copy(),'validation')
            #save config
            with open(Path(full_directory.parent,'val','config.json'),'w') as f:
                json.dump(data_parameters, f)
            generate_dataset(data_parameters, Path(full_directory.parent, 'val'))

            #test data
            data_parameters=mod_task_parameters(data_parameters,'test')
            #save config
            with open(Path(full_directory.parent,'test','config.json'),'w') as f:
                json.dump(data_parameters, f)
            generate_dataset(data_parameters, Path(full_directory.parent, 'test'))
        super().__init__(full_directory,file_extension, data_parameters,device=device)

    def load_file(self, file_path):
        data = pd.read_csv(file_path)
        return data

    def get_channels(self):
        time='time'
        x_channels = []
        if self.data_parameters["System"] == 'DampedOscillator':
            y_channels = ['position','velocity']
        if self.data_parameters["System"] == 'LowPassFilter':
            y_channels = ['y']
        if self.data_parameters["System"] == 'LorenzSystem':
            y_channels = ['x','y','z']

        z_channels=[]
        return [time,x_channels,y_channels,z_channels]

def mod_task_parameters(data_parameters,dataset_type):
    """
    Modify the task parameters based on the task.

    Parameters:
    data_parameters (dict): The data parameters.

    Returns:
    dict: The modified data parameters.
    """
    task = data_parameters["Task"]
    task_parameters = data_parameters["TaskParameters"]
    if task == "Reconstruction":
        data_parameters["TaskParameters"]["initial_values_range_validation"] = task_parameters.get(
            "initial_values_range_validation", [-1, 1]
        )
        data_parameters["TaskParameters"]["initial_values_range_test"] = task_parameters.get(
            "initial_values_range_test", [-1, 1]

        )
        #change the seed +1 for the next task
        data_parameters["seed"] += 1
    elif task == "Extrapolation":
        if dataset_type == 'validation':
            data_parameters["final_time"] = data_parameters["TaskParameters"]["final_time_validation"]
        elif dataset_type == 'test':
            data_parameters["final_time"] = data_parameters["TaskParameters"]["final_time_test"]

    elif task == "Interpolation":
        if dataset_type == 'validation':
            data_parameters["time_step_simulation"] = data_parameters["TaskParameters"]["step_time_validation"]
        elif dataset_type == 'test':
            data_parameters["time_step_simulation"] = data_parameters["TaskParameters"]["step_time_test"]
    return data_parameters




def generate_hash_from_dict(param_dict):
    """
    Generate a hash from a dictionary.

    Parameters:
    param_dict (dict): The dictionary to hash.

    Returns:
    str: The hash of the dictionary.
    """
    sorted_dict_string = json.dumps(param_dict, sort_keys=True)
    hash_object = hashlib.sha1(sorted_dict_string.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex


def generate_dataset(config,path=None):
    """
    Generate a dataset based on the provided configuration.

    Parameters:
    config (dict): The configuration dictionary.

    Returns:
    list: A list of samples generated based on the configuration.
    """



    np.random.seed(config["seed"])

    samples = []

    for _ in range(config["n_samples"]):
        sample = generate_single_sample(config)
        samples.append(pd.DataFrame(sample))

    if path is not None:
        #save as csv.gz (pandas)
        for i, sample in enumerate(samples):
            sample.to_csv(f"{path}/sample_{i}.csv.gz", index=False, compression='gzip')

    return samples


def generate_single_sample(config):
    """
    Generate a single sample for the  dataset.

    Parameters:
    config (dict): The configuration dictionary.

    Returns:
    dict: A sample containing time, temperature, and command values.
    """

    if config["System"] == 'DampedOscillator':
        f=DampedOscillator
        dim_y = 2

    if config["System"] == 'LowPassFilter':
        f=LowPassFilter
        dim_y = 1

    if config["System"] == 'LorenzSystem':
        f=lorenz_system
        dim_y = 3


    initial_value = np.random.uniform(*config["inital_values_range"],dim_y)

    t = np.arange(0, config["final_time"], config["time_step_simulation"])

    param=config["SystemParameters"]

    if config["System"] == 'LorenzSystem':
        monitors=list('xyz')
        def f_wrapper(x, y, z, t):
            return f(x, y, z, t,**param)
    if config["System"] == 'DampedOscillator':
        monitors=['y1','y2','y3','y4']
        initial_value=np.array([initial_value[0],0,initial_value[1],0])
        def f_wrapper(y1,y2,y3,y4,t):
            return f(y1,y2,y3,y4,t, **param)
    if config["System"] == 'LowPassFilter':
        monitors=['y']
        def f_wrapper(y ,t):
            return f(t, y, **param)

    #use numpy to solve the FDE
    alpha=config["alpha"]
    integrator=bp.fde.CaputoL1Schema(f_wrapper,
                          alpha=alpha,  # fractional order
                          num_memory=int(config["final_time"] / config["time_step_simulation"]),
                          inits=initial_value.tolist())
    runner = bp.IntegratorRunner(integrator,
                                 monitors=monitors,
                                 inits=initial_value.tolist(),
                                 dt=config["time_step_simulation"],
                                 progress_bar=False)
    runner.run(config["final_time"])

    res = runner.mon


    #t has to stat at 0
    res.ts=res.ts-res.ts[0]

    if config["System"] == 'LorenzSystem':
        sample = {
            "time": res.ts,
            "x": res.x.flatten()/35,
            "y": res.y.flatten()/25,
            "z": res.z.flatten()/15}
    if config["System"] == 'DampedOscillator':
        sample = {
            "time": res.ts,
            "position": res.y1.flatten(),
            "velocity": res.y3.flatten()}

    if config["System"] == 'LowPassFilter':

        sample = {
            "time": res.ts,
            "y": res.y.flatten()}




    return sample

def DampedOscillator(y1,y2,y3,y4,t, m, dc, sc,f):
    """
    The ODE system for a damped oscillator.

    Parameters:
    t (float): The time.
    y1 (float): The position.
    y2 (float): Internal variable 1.
    y3 (float): The velocity.
    y4 (float): Internal variable 2.
    m (float): The mass.
    dc (float): The damping coefficient.
    sc (float): The spring constant.
    f (float): The applied force.

    Returns:
    d_y1 (float): The derivative of the position.
    d_y2 (float): The derivative of the internal variable 1.
    d_y3 (float): The derivative of the velocity.
    d_y4 (float): The derivative of the internal variable2 .
    """

    d_y1 = y2
    d_y2 = y3
    d_y3 = y4
    d_y4 = (1 / m) * (-sc * y1 - dc * y4 + f)
    return d_y1, d_y2, d_y3, d_y4

def LowPassFilter(t, y, Tc, final_value):
    """
    The ODE system for a low-pass filter.

    Parameters:
    t (float): The time.
    y (np.array): The state of the system.
    Tc (float): The time constant.
    final_value (float): The final value of the system.

    Returns:
    np.array: The derivative of the state.
    """
    y=-y[0]/Tc + final_value
    return y

def lorenz_system(x, y, z, t,a,b,c):
  dx = a * (y - x)
  dy = x * (b - z) - y
  dz = x * y - c * z
  return dx, dy, dz

if __name__ == "__main__":
    config = {
            "n_samples": 5,
            "System": 'LorenzSystem',
            "Task": "Reconstruction",
            "TaskParameters": {
                "initial_values_range_validation": [-1.2, 1.2],
                "initial_values_range_test": [-2, 2],
            },
            "SystemParameters": {
                "a": 8,
                "b": 10,
                "c": 1.5
            },
            "alpha": 0.96,
            "noise": 0.0,
            "inital_values_range": [-2, 2],
            "time_step_simulation": 0.005,
            "final_time": 3,
            "seed": 42
        }

    configOS = {
        "n_samples": 2,
        "System": 'DampedOscillator',
        "Task": "Reconstruction",
        "TaskParameters": {
            "initial_values_range_validation": [-2, 2],
            "initial_values_range_test": [-1.1, 1.1],
        },
        "SystemParameters": {
            "m": 2.5,  # mass
            "dc": 0.75,  # damping coefficient
            "sc": 10,  # spring constant
            "f":0 # applied force
        },
        "noise": 0.0,
        "alpha": 0.5,
        "inital_values_range": [-1, 1],
        "time_step_simulation": 0.00025,
        "final_time": 20,
        "seed": 42
    }

    configLP = {
        "n_samples": 2,
        "System": 'LowPassFilter',
        "Task": "Reconstruction",
        "TaskParameters": {
            "initial_values_range_validation": [-1, 1],
            "initial_values_range_test": [-1, 1],
        },
        "SystemParameters": {
            "Tc": 2,  # time constant
            "final_value": 0  # final value
        },
        "noise": 0.0,
        "alpha": 0.9,
        "inital_values_range": [-1, 1],
        "time_step_simulation": 0.01,
        "final_time": 20,
        "seed": 42
    }

    configLP_extrapol = {
        "n_samples": 20,
        "System": 'LowPassFilter',
        "Task": "Extrapolation",
        "TaskParameters": {
            "final_time_validation": 3,
            "final_time_test": 10,
        },
        "SystemParameters": {
            "Tc": 2,  # time constant
            "final_value": 0  # final value
        },
        "noise": 0.0,
        "alpha": 0.8,
        "inital_values_range": [-1, 1],
        "time_step_simulation": 0.1,
        "final_time": 1.5,
        "seed": 42
    }
    configOS_extrap = {
        "n_samples": 20,
        "System": 'DampedOscillator',
        "Task": "Extrapolation",
        "TaskParameters": {
            "final_time_validation": 8.25,
            "final_time_test": 15,
        },
        "SystemParameters": {
            "m": 1,  # mass
            "dc": 0.6,  # damping coefficient
            "sc": 10,  # spring constant
            "f": 0  # applied force
        },
        "noise": 0.0,
        "alpha": 0.5,
        "inital_values_range": [-1, 1],
        "time_step_simulation": 0.005,
        "final_time": 7.5,
        "seed": 42
    }

 

    #test the dataset class

    dataset = HomogeneousFDEDataset(data_parameters=configOS_extrap,directory='test')

    ##### go over desied amout of samples and convert them to dataframe

    samples_plot =-1 #-1 is all

    if samples_plot == -1:
        samples_plot = len(dataset)
    samples = []
    for i in range(min(samples_plot,len(dataset))):
        samples.append(dataset[i])

    ch_names_y=dataset.get_channels()[2]

    #plot the samples
    fig=go.Figure()
    for j,sample in enumerate(samples):
        time=sample[0].cpu().numpy()
        y=sample[2].cpu().numpy()
        for i in range(y.shape[1]):
            fig.add_trace(go.Scatter(x=time, y=y[:,i], mode='lines', name='Sample_'+str(j)+'_'+ch_names_y[i]))
    fig.show()





