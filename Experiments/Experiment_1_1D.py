from pathlib import Path
import os
import sys
import pprint
PROJECT_PATH = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(Path(PROJECT_PATH)))


import torch
import numpy as np
import pandas as pd
import json

from torch.utils.data import DataLoader
from torch.optim import Adam
from src.datasets.HomogeneousFDE import HomogeneousFDEDataset
from src.models.NFDE import FDE_Model
from src.models.NODE import NODE_Model
import Constants as const
from ray import tune,train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

import ray
#seeds etc for reproducibility
torch.manual_seed(42)
np.random.seed(42)
pd.options.mode.chained_assignment = None
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False






# Seed setup for reproducibility
def setup_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration management
def create_config(model_type='NODE', FLAG_TEST=False, Dataset='DampedOscillatorReconstruction_without_time'):
    """
    Create configuration for the experiment
    :param model_type: Type of the model
    """
    # Base configuration setup
    config = {
        'data': {'dataset': Dataset},
        'training': {'learning_rate': 0.001, 'batch_size': 100, 'epochs': 100, 'optimizer': 'Adam', 'gamma_scheduler': 0.9999},
        'model': {}
    }

    if FLAG_TEST:
        config['training']['epochs'] = 1
        config['training']['epochs_pretraining'] = 1

    if 'with_time' in Dataset:
        include_time = True
    elif 'without_time' in Dataset:
        include_time = False
    else:
        raise ValueError(f"Dataset {Dataset} not recognized.")

    if 'DampedOscillator' in Dataset:
        if 'ODE' in Dataset:
            config['training']['epochs'] = 200
            config['data'].update({'dim_x': 0, 'dim_y': 2})
            task_type = 'Reconstruction' if 'Reconstruction' in Dataset else 'Extrapolation'
            if task_type == 'Reconstruction':
                task_param = {
                    'initial_values_range_validation': [-1, 1],
                    'initial_values_range_test': [-1, 1]
                }
            elif task_type == 'Extrapolation':
                task_param = {
                    'final_time_validation': 8.25,
                    'final_time_test': 15
                }
            else:
                raise ValueError(f"Task {task_type} not recognized.")
            data_config = {
                "n_samples": 5,
                "ODESystem": 'DampedOscillator',
                "Task": task_type,
                "TaskParameters": task_param,
                "SystemParameters": {
                    "m": 1,  # mass
                    "dc": 0.1,  # damping coefficient
                    "sc": 1  # spring constant
                },
                "noise": 0.0,
                "inital_values_range": [-1, 1],
                "time_step_simulation": 0.1,
                "final_time": 20 if task_type == 'Reconstruction' else 7.5,
                "seed": 42
            }
        elif 'FDE' in Dataset:
            config['training']['epochs'] = 150
            config['data'].update({'dim_x': 0, 'dim_y': 2})
            task_type = 'Reconstruction' if 'Reconstruction' in Dataset else 'Extrapolation'
            if task_type == 'Reconstruction':
                task_param = {
                    'initial_values_range_validation': [-2, 2],
                    'initial_values_range_test': [-1.1, 1.1]
                }
            elif task_type == 'Extrapolation':
                task_param = {
                    'final_time_validation': 8.25,
                    'final_time_test': 15
                }
            else:
                raise ValueError(f"Task {task_type} not recognized.")
            data_config = {
                "n_samples": 80,
                "System": 'DampedOscillator',
                "Task": task_type,
                "TaskParameters": task_param,
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
                "final_time": 15 if task_type == 'Reconstruction' else 7.5,
                "seed": 42
            }
        else:
            raise ValueError(f"Dataset {Dataset} not recognized.")

    elif 'LowPassFilter' in Dataset:
        if 'ODE' in Dataset:
            config['training']['epochs'] = 50
            config['data'].update({'dim_x': 0, 'dim_y': 1})
            task_type = 'Reconstruction' if 'Reconstruction' in Dataset else 'Extrapolation'
            if task_type == 'Reconstruction':
                task_param = {
                    'initial_values_range_validation': [-1, 1],
                    'initial_values_range_test': [-1, 1]
                }
            elif task_type == 'Extrapolation':
                task_param = {
                    'final_time_validation': 3,
                    'final_time_test': 10
                }
            else:
                raise ValueError(f"Task {task_type} not recognized.")
            data_config = {
                "n_samples": 5,
                "ODESystem": 'LowPassFilter',
                "Task": task_type,
                "TaskParameters": task_param,
                "SystemParameters": {
                    "Tc": 2,  # time constant
                    "final_value": 0  # final value
                },
                "noise": 0.0,
                "inital_values_range": [-1, 1],
                "time_step_simulation": 0.1,
                "final_time": 20 if task_type == 'Reconstruction' else 1.5,
                "seed": 42
            }
        elif 'FDE' in Dataset:
            config['training']['epochs'] = 50
            config['data'].update({'dim_x': 0, 'dim_y': 1})
            task_type = 'Reconstruction' if 'Reconstruction' in Dataset else 'Extrapolation'
            if task_type == 'Reconstruction':
                task_param = {
                    'initial_values_range_validation': [-2, 2],
                    'initial_values_range_test': [-1.5, 1.5]
                }
            elif task_type == 'Extrapolation':
                task_param = {
                    'final_time_validation': 3,
                    'final_time_test': 10
                }
            else:
                raise ValueError(f"Task {task_type} not recognized.")
            data_config = {
                "n_samples": 25,
                "System": 'LowPassFilter',
                "Task": task_type,
                "TaskParameters": task_param,
                "SystemParameters": {
                    "Tc": 2,  # time constant
                    "final_value": 0  # final value
                },
                "noise": 0.0,
                "alpha": 0.8,
                "inital_values_range": [-1, 1],
                "time_step_simulation": 0.1,
                "final_time": 20 if task_type == 'Reconstruction' else 1.5,
                "seed": 42
            }
        else:
            raise ValueError(f"Dataset {Dataset} not recognized.")

    elif 'BloodClot' in Dataset:
        data_config = {}
        config['data'].update({'dim_x': 1, 'dim_y': 1})
    elif 'LorenzSystem' in Dataset:
        config['data'].update({'dim_x': 0, 'dim_y': 3})
        data_config = {
            "n_samples": 250,
            "System": 'LorenzSystem',
            "Task": "Reconstruction",
            "TaskParameters": {
                "initial_values_range_validation": [-1, 1],
                "initial_values_range_test": [-1.25, 1.25],
            },
            "SystemParameters": {
                "a": 8,
                "b": 10,
                "c": 1.5
            },
            "alpha": 0.96,
            "noise": 0.0,
            "inital_values_range": [-1, 1],
            "time_step_simulation": 0.005,
            "final_time": 3,
            "seed": 42
        }
    else:
        raise ValueError(f"Dataset {Dataset} not recognized.")

    config['model'].update({
        'include_time': include_time,
        'homogeneous': True,
    })

    config['data'].update({'params': data_config})

    return config

# Data loading
def setup_data_loaders(device, config,num_workers=0):
    dataset_cls=get_ds_class(config['data']['dataset'])
    datasets = {x: dataset_cls(directory=x, device=device, config=config['data']) for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(datasets[x],
                                 batch_size=config['training']['batch_size'],
                                 shuffle=(x == 'train'),
                                 collate_fn=dataset_cls.collate_fn,
                                 num_workers=num_workers) for x in ['train', 'val', 'test']}
    return dataloaders

def get_ds_class(dataset_name):

    if 'LowPassFilter' in dataset_name:
        if 'FDE' in dataset_name:
            return HomogeneousFDEDataset
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")

# Model setup
def setup_model(model_type, config):
    assert model_type in ['NODE', 'FDE', 'Laplace','FDE_fix_alpha','FDE_augmented','NODE_augmented']
    if model_type == 'NODE':
        config['model'].update({'augmented_dim': 0, 'augmentation_type': 'none'})
        #solver fixed
        config['model'].update({'solver': 'explicit_adams'})
        model = NODE_Model(config)
    elif model_type == 'FDE':
        config['model'].update({'augmented_dim': 0, 'augmentation_type': 'none'})
        model = FDE_Model(config)
    elif model_type == 'FDE_fix_alpha':
        config['model'].update({'augmented_dim': 0, 'augmentation_type': 'none'})
        model = FDE_Model(config)
    elif model_type == 'FDE_augmented':
        model = FDE_Model(config)
    elif model_type == 'NODE_augmented':
        config['model'].update({'solver': 'explicit_adams'})
        model = NODE_Model(config)
    elif model_type == 'Laplace':
        model = Laplace_Model(config)
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    return model,config


def setup_searchspace(model_type,dataset):
    search_space = {'training':
                        {'learning_rate': tune.loguniform(1e-5, 5e-1),
                        #'batch_size': tune.choice([10,20,30,40,50]),
                        'gamma_scheduler':tune.uniform(0.999,1),
                         },
                    'model':{}
                    }
    if dataset=='BloodClot':
        #we can only use batch size 1
        search_space['training']['batch_size']=tune.choice([1])

    if model_type == 'NODE':
        search_space['model'].update(
            {'hidden_size_ODE': tune.randint(10, 101),
             'n_layers_hidden_ODE': tune.randint(1, 5),
             'activation_ODE': tune.choice(['relu', 'leaky_relu', 'elu','none'])
             }
        )
    elif model_type == 'NODE_augmented':
        search_space['model'].update(
            {'hidden_size_ODE': tune.randint(10, 101),
             'n_layers_hidden_ODE': tune.randint(1, 5),
             'activation_ODE': tune.choice(['relu', 'leaky_relu', 'elu','none']),
             'augmented_dim': tune.randint(1, 5),
             'augmentation_type': tune.choice(['zeros'])
             }
        )
    elif model_type == 'FDE':
        search_space['model'].update(
            {'hidden_size_FDE': tune.randint(10, 101),
             'n_layers_hidden_FDE': tune.randint(1, 5),
             'activation_FDE': tune.choice(['relu', 'leaky_relu', 'elu','none']),
             'alpha': tune.uniform(0.5, 1.0)
             }
        )
    elif model_type == 'FDE_fix_alpha':
        search_space['model'].update(
            {'hidden_size_FDE': tune.randint(10, 101),
             'n_layers_hidden_FDE': tune.randint(1, 5),
             'activation_FDE': tune.choice(['relu', 'leaky_relu', 'elu','none']),
             }
        )
    elif model_type == 'FDE_augmented':
        search_space['model'].update(
            {'hidden_size_FDE': tune.randint(10, 101),
             'n_layers_hidden_FDE': tune.randint(1, 5),
             'activation_FDE': tune.choice(['relu', 'leaky_relu', 'elu','none']),
             'alpha': tune.uniform(0.25, 0.75),
             'augmented_dim': tune.randint(1, 5),
             'augmentation_type': tune.choice(['zeros'])
             }
        )
    elif model_type == 'Laplace':
        search_space['model'].update(
            {'hidden_size_Laplace': tune.randint(10, 101),
             'n_layers_hidden_Laplace': tune.randint(1, 5),
            'activation_LPF': tune.choice(['relu', 'leaky_relu', 'tanh', 'elu']),
             'ilt_reconstruction_terms': tune.choice([11, 21, 31, 41, 51, 61, 71, 81, 91, 101])# odd number of terms in the reconstruction
             }
        )
    return search_space

# Training function
def train_model(model, dataloaders, device, config):
    optimizer = Adam(model.parameters(), lr=config['training']['learning_rate'])
    # if config['training']['epochs_pretraining'] > 0:
    #     pretrain_model(model, dataloaders['train'], optimizer, config['training']['epochs_pretraining'])
    losses_df = pd.DataFrame(columns=['epochs','train', 'val','test'])
    #check losses after pretraining
    model.eval()
    loss_train = model.train_step(dataloaders['train'], return_loss=True)
    loss_val = model.validate_step(dataloaders['val'])
    loss_test = model.validate_step(dataloaders['test'])
    losses_df.loc[0] = [0,loss_train,loss_val,loss_test]
    for epoch in range(1,config['training']['epochs']):
        model.train()
        train_loss = model.train_step(dataloaders['train'], return_loss=True)
        model.eval()
        val_loss = model.validate_step(dataloaders['val'])
        test_loss = model.validate_step(dataloaders['test'])
        losses_df.loc[epoch] = [epoch,train_loss,val_loss,test_loss]
    return losses_df
def recursive_update(base_dict, new_dict):
    """
    Recursively updates base_dict with values from new_dict. If both base_dict and new_dict
    contain a dictionary at a given key, then it merges those dictionaries via recursive call.
    Otherwise, it updates the value in base_dict with the value in new_dict.
    """
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value

    return base_dict
def train_hparams(config, base_config,model_type, device):
    # Merge hyperparameter tuning configurations with base configurations

    new_config = recursive_update(base_config, config)
    dataloaders =setup_data_loaders(device, new_config)
    model,new_config= setup_model(model_type, new_config)
    model.to(device)


    for epoch in range(new_config['training']['epochs']):
        model.train()
        loss_train = model.train_step(dataloaders['train'])

        model.eval()
        loss_val = model.validate_step(dataloaders['val'])
        # Report metrics to Ray Tune
        train.report(metrics={"loss": loss_val, "train_loss": loss_train,'epoch':epoch})



    # This should return the metric value that needs to be minimized/maximized
    return {"loss": loss_val}



# Hyperparameter tuning
def tune_hyperparameters(base_config, model_type, device, num_samples=25, seed=42):
    # check if we want to debug or using the debugger
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        debugger = True
        print('Ray enters debugging mode using a single local instance')
    else:
        debugger = False

    ray.init(local_mode=(debugger or FLAG_TEST))

    search_space = setup_searchspace(model_type, base_config['data']['dataset'])

    if FLAG_TEST:
        num_samples = 1
        base_config['training']['epochs'] = 1

    if device.type == 'cuda':
        num_gpus = 0.5
    else:
        num_gpus = 0

    # Configure the scheduler and search algorithm
    scheduler = ASHAScheduler(
        max_t=base_config['training']['epochs'] + 1,  # Maximum number of training iterations (epochs)
        # Minimum number of epochs to run before stopping poorly performing trials
        grace_period=min(base_config['training']['epochs'] + 1,
                         max(10,
                             max(int(base_config['training']['epochs'] // 2),
                                 1)
                             )
                         ),
        reduction_factor=3)
    search_alg = OptunaSearch(seed=seed)
    trainable = tune.with_resources(train_hparams, {"cpu": 1, "gpu": num_gpus})
    trainable = tune.with_parameters(trainable, base_config=base_config, model_type=model_type, device=device)

    # Create the tuner and start the tuning process
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            num_samples=num_samples,
            scheduler=scheduler,
            metric="loss",
            mode="min",
        ),
        param_space=search_space,
    )

    analysis = tuner.fit()
    best_trial = analysis.get_best_result("loss", "min", "all")
    print('Hyperparameter tuning for model {} with seed {} has finished.'.format(model_type, seed))
    print(analysis.num_terminated, "have been completed.", analysis.num_errors, "trails have errored out.")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}  after {} epochs".format(best_trial.metrics['loss'], best_trial.metrics['training_iteration'] - 1))

    # New code for detailed statistics
    all_trials = analysis._results
    trial_statistics = []
    for trial in all_trials:
        trial_statistics.append({
            "config": trial.config,
            "loss": trial.metrics['loss'],
            "iterations": trial.metrics['training_iteration'],
            "error": trial.error,
            "path": trial.path,
        })

    # Log the number of failed trials
    num_errors = analysis.num_errors

    aux_results = analysis.get_dataframe()
    ray.shutdown()

    best_hyperparameters = recursive_update(base_config, best_trial.config)
    return best_hyperparameters, best_trial.metrics, aux_results, best_trial.config, trial_statistics, num_errors

# Run experiments
def run_experiments(name_models,device,seeds,path,dataset,hparam_trails=100,search_hparams=True):

    results={model : {} for model in name_models}
    for name_model in name_models:
        results[name_model]={seed : {'results':{},'num_errors':{}} for seed in seeds}
        for seed in seeds:
            setup_seeds(seed)
            config=create_config(model_type=name_model,FLAG_TEST=FLAG_TEST,Dataset=dataset)
            dataloaders = setup_data_loaders(device,
                                             config)#make sure ds is create before ray starts
            path_hparams = Path(path, f'{name_model}_seed_{seed}')
            #check if we want to search for hyperparameters // there are already hyperparameters in the config
            if Path(path_hparams, 'config.json').exists() and not search_hparams:
                with open(Path(path_hparams, 'config.json'), 'r') as f:
                    full_config = json.load(f)
                num_errors = 0
            else:

                full_config, metrics, aux_results, hyperparameters, trial_statistics, num_errors= tune_hyperparameters(
                                                                                            config,
                                                                                            name_model,
                                                                                            device,
                                                                                            num_samples=hparam_trails,
                                                                                            seed=seed
                                                                                            )
                #save the hyperparameters and results

                path_hparams.mkdir(parents=True,exist_ok=True)
                if FLAG_TEST:
                    print('Hyperparameters:',hyperparameters)
                    print('Metrics:',metrics)
                    print('Auxiliary results:',aux_results)
                    print('Full config:',full_config)
                    print('Trial statistics:',trial_statistics)
                    print('Number of errors:',num_errors)
                else:
                    with open(Path(path_hparams,'hyperparameters.json'),'w') as f:
                        json.dump(hyperparameters,f)
                    with open(Path(path_hparams,'config.json'),'w') as f:
                        json.dump(full_config,f)
                    with open(Path(path_hparams,'metrics.json'),'w') as f:
                        json.dump(metrics,f)
                    with open(Path(path_hparams,'trial_statistics.json'),'w') as f:
                        json.dump(trial_statistics,f)
                    with open(Path(path_hparams,'num_errors.json'),'w') as f:
                        json.dump(num_errors,f)
                    aux_results.to_csv(Path(path_hparams,'aux_results.csv'))


            #if cpu, only use 1 worker
            if device.type=='cpu':
                   torch.set_num_threads(4)
            #for the final trainge we use more epochs
            full_config['training']['epochs']=200

            #train the model with the best hyperparameters
            dataloaders = setup_data_loaders(device,
                                             full_config)  # this is done again inside the hyperparameter tuning as ray tune does not 'like' big objects to be passed
            model,full_config=setup_model(name_model,full_config)
            model.to(device)
            res_df=train_model(model, dataloaders, device, full_config)
            try:
                model.save_model(str(Path(path_hparams,'model.pth')))
            except:
                print('Model could not be saved')
            results[name_model][seed]['results']=res_df
            results[name_model][seed]['num_errors']=num_errors
            #save the results
            res_df.to_csv(Path(path_hparams,'results_training.csv'))
    return results

# Main entry point
if __name__ == "__main__":
    # check if cuda is available
    device = torch.device('cpu') #the datasets are too small to use a GPU
    #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    FLAG_TEST = False
    seeds =[0, 8,42,98]
    name_models=['NODE','FDE',]
    name_datasets=['FDELowPassFilterReconstruction_with_time',
                   'FDELowPassFilterExtrapolation_with_time',
                   'FDELowPassFilterReconstruction_without_time',
                   'FDELowPassFilterExtrapolation_without_time']


    name_experiment = Path(__file__).stem
    hparam_trails=1000
    search_hparams=True

    for dataset in name_datasets:
        print('Processing dataset:',dataset)
        path = Path(const.RESULTS_PATH, name_experiment,dataset)
        path.mkdir(parents=True, exist_ok=True)
        #run the experiments
        results = run_experiments(name_models, device, seeds, path,dataset,hparam_trails=hparam_trails,search_hparams=search_hparams)


        # go through all results and mean them over the seeds, return the mean and std in dataframes
        errors = {}
        results_mean_std = {}
        for name_model in name_models:
            results_mean_std[name_model] = {}
            #aggregate the all dataframes for each model
            for key in results[name_model][seeds[0]]['results'].keys():
                results_mean_std[name_model][key] = pd.concat([results[name_model][seed]['results'][key] for seed in seeds]).groupby(level=0).agg(['mean', 'std'])
            errors[name_model] = {seed: results[name_model][seed]['num_errors'] for seed in seeds}

        # save the results
        for name_model in name_models:
            for key in results_mean_std[name_model].keys():
                results_mean_std[name_model][key].to_csv(Path(path, f'{name_model}_{key}_mean_std.csv'))
            # save the errors
        with open(Path(path, 'errors.json'), 'w') as f:
            json.dump(errors, f)
        print('Finished processing dataset:',dataset)

        #summary of the errors for the models
        print('Errors for the models')
        pprint.pprint(errors)












