# Optimising Neural Fractional Differential Equations for Performance and Efficiency

This repository contains the code for the paper titled *"Optimising Neural Fractional Differential Equations for Performance and Efficiency"* by Bernd Zimmering, Cecília Coelho, and Oliver Niggemann. The paper is part of the proceedings for the 1st ECAI Workshop on *"Machine Learning Meets Differential Equations: From Theory to Applications"*.

You can access the paper in the following ways:
- View the [official PMLR publication](https://proceedings.mlr.press/v255/zimmering24a.html).
- Download the local version of the paper [here](zimmering24a.pdf).


## Abstract
Neural Ordinary Differential Equations (NODEs) are well-established architectures that fit an ODE, modelled by a neural network (NN), to data, effectively modelling complex dynamical systems. Recently, Neural Fractional Differential Equations (NFDEs) were proposed, inspired by NODEs, to incorporate non-integer order differential equations, capturing memory effects and long-range dependencies. In this work, we present an optimised implementation of the NFDE solver, achieving up to 570 times faster computations and up to 79 times higher accuracy. Additionally, the solver supports efficient multidimensional computations and batch processing. Furthermore, we enhance the experimental design to ensure a fair comparison of NODEs and NFDEs by implementing rigorous hyperparameter tuning and using consistent numerical methods. Our results demonstrate that for systems exhibiting fractional dynamics, NFDEs significantly outperform NODEs, particularly in extrapolation tasks on unseen time horizons. Although NODEs can learn fractional dynamics when time is included as a feature to the NN, they encounter difficulties in extrapolation due to reliance on explicit time dependence.

## Table of Contents
- [Abstract](#abstract)
- [Python Version & Required Packages](#python-version--required-packages)
- [Code Structure](#code-structure)
- [Dataset](#dataset)
- [Experiments](#experiments)
- [Usage](#usage)
- [Questions and Issues](#questions-and-issues)
- [Citation](#citation)
- [License](#license)

## Python Version & Required Packages
- Python 3.11
- Used Packages:
```bash
pytorch 2.2.1
torchdiffeq 0.2.3
matplotlib 3.8.0
pandas 2.2.1
scikit-learn 1.3.0
tqdm 4.66.2
plotly 5.20.0
torchlaplace 0.0.4
ray[data,train,tune,serve] 2.23.0
optuna 3.6.1 
brainpy[cpu] 2.6.0
memory-profiler 0.61.0
psutil 6.0.0
```

Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
## Code Structure

- src: Main source code, including the optimized solver.
- experiments: Contains all experiment scripts for running tests on both NODEs and NFDEs.
- models: Defines the NODE and NFDE architectures.
- utils: Helper functions for data processing, plotting, and post-processing.
- data: Includes the generated datasets for training, validation, and testing.
- results: Stores the results of the experiments, including logs and plots.

## Dataset

All datasets are generated and stored in the data folder, divided into train, validation, and test sets. We include example datasets with the necessary configurations for fractional and non-fractional systems.

## Experiments

This repository supports two main experiments:

1. **Solver Performance Evaluation**: This experiment compares the performance of the optimized NFDE solver with the original implementation, focusing on computation speed and accuracy improvements.
  
2. **Comparison of NODE and NFDE**: This experiment evaluates the performance of Neural Ordinary Differential Equations (NODEs) and Neural Fractional Differential Equations (NFDEs) on a system with fractional dynamics. It includes two tasks:
   - **Extrapolation Task**: Testing how well the models predict unseen time horizons.
   - **Reconstruction Task**: Assessing the models' ability to reconstruct time-series data for systems exhibiting sub-exponential growth.

For the comparison experiment, hyperparameter tuning is performed using a Tree-Parzen Estimator with multiple seeds to ensure robust and fair comparisons. All results are stored in the results folder.

## Usage

To run an experiment, navigate to the experiments folder and execute the desired script. For instance:



Ensure that the src folder is in your Python path if necessary:

- **Windows:**

```bash
set PYTHONPATH=%PYTHONPATH%;%CD%
```

- **Linux/macOS:**

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Questions and Issues
If you encounter any issues or have questions, please open an issue in this repository, or reach out to bernd.zimmering@hsu-hh.de.

## Citation
If you find this work useful for your research, please consider citing the paper:
```
@InProceedings{pmlr-v255-zimmering24a,  
title = {Optimising Neural Fractional Differential Equations for Performance and Efficiency},  
author = {Zimmering, Bernd and Coelho, Cec\'{i}lia and Niggemann, Oliver},  
booktitle = {Proceedings of the 1st ECAI Workshop on "Machine Learning Meets Differential Equations: From Theory to Applications"},  
pages = {1--22},  
year = {2024},  
editor = {Coelho, Cecı́lia and Zimmering, Bernd and Costa, M. Fernanda P. and Ferrás, Luı́s L. and Niggemann, Oliver},  
volume = {255},  
series = {Proceedings of Machine Learning Research},  
month = {20 Oct},  
publisher = {PMLR},  
pdf = {https://raw.githubusercontent.com/mlresearch/v255/main/assets/zimmering24a/zimmering24a.pdf},  
url = {https://proceedings.mlr.press/v255/zimmering24a.html},  
abstract = {Neural Ordinary Differential Equations (NODEs) are well-established architectures that fit an ODE, modelled by a neural network (NN), to data, effectively modelling complex dynamical systems. Recently, Neural Fractional Differential Equations (NFDEs) were proposed, inspired by NODEs, to incorporate non-integer order differential equations, capturing memory effects and long-range dependencies. In this work, we present an optimised implementation of the NFDE solver, achieving up to 570 times faster computations and up to 79 times higher accuracy. Additionally, the solver supports efficient multidimensional computations and batch processing. Furthermore, we enhance the experimental design to ensure a fair comparison of NODEs and NFDEs by implementing rigorous hyperparameter tuning and using consistent numerical methods. Our results demonstrate that for systems exhibiting fractional dynamics, NFDEs significantly outperform NODEs, particularly in extrapolation tasks on unseen time horizons. Although NODEs can learn fractional dynamics when time is included as a feature to the NN, they encounter difficulties in extrapolation due to reliance on explicit time dependence. The code is available at https://github.com/zimmer-ing/Neural-FDE}}
```
or if you directly want to cite this repository:
```
@misc{zimmering_neural_fde_2024,
    author       = {Bernd Zimmering and Cec\'{\i}lia Coelho and Oliver Niggemann},
    title        = {Code for Optimising Neural Fractional Differential Equations for Performance and Efficiency},
    year         = {2024},
    url          = {https://github.com/zimmer-ing/Neural-FDE},
    note         = {GitHub repository},
    howpublished = {\url{https://github.com/zimmer-ing/Neural-FDE}}
}
```