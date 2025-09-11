# Graph-based metamodeling (GraphMM) to uncover cell dynamics and function across molecular, cellular, and multicellular scales

This is the official repository for GraphMM, a Python package to uncover cell dynamics and function across molecular, cellular, and multicellular scales.

# Introduction

We introduce Graph-based Metamodeling (GraphMM), a novel framework that integrates models across multiple representations and spatiotemporal scales, by (i) converting input models into universal surrogate representations using probabilistic graphical models; (ii) coupling surrogates across time scales using a standardized strategy; and (iii) approximate metamodel inference. Validation through synthetic benchmarks and real-world applications shows improved accuracy over existing methods. GraphMM enables quantitative predictions of $\beta$-cell dynamics and function across molecular, cellular, and multicellular scales. GraphMM provides a versatile framework for integrating models to uncover the dynamics of complex systems. 
For detailed documentation, please visit: https://graphmm.readthedocs.io/

<p align="center">
  <img src="./GraphMM.png" width="800"/>
</p>


# Repo Structure

The project contains a benchmark and a Multiscale β-cell metamodel (MuBCM), both using the GraphMM modeling framework. The project structure is as follows:
1. `Benchmark/`: Contains the benchmark toy system for GraphMM using a toy GSIS model.  
**Quick start**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mmF2elmdj9g20Y7XFUviKmAUEDBGiv6K?usp=sharing)
   

2. `GraphMM_MuBCM/`: The main package for GraphMM
     - `InputModel/`:
        - Contains subsystem models 
    - `GraphMetamodel/`:
        - Defines connections between surrogate models
        - Implements multi-scale inference
    - `results/`:
        - Stores output files from model simulations   



# Usage

### Prerequisite

```bash
python           3.7
scikit-learn     1.0.2
pandas           1.3.5
numpy            1.21.6
scipy            1.7.3
filterpy         1.4.5
daft             2.10.0
matplotlib       3.5.3
jupyter          1.0.0
ipykernel        6.15.2
```

### Benchmark - Toy GSIS metamodel
1. To run the metamodel enumeration:

```bash
cd Benchmark
python Surrogate_model_a.py
python Surrogate_model_b.py
```

2. Results will be saved in the `results/` directory
3. Visualizations can be generated using the plotting functions in the script

```bash
python visulize.py
```



### Multiscale β-cell metamodel (MuBCM)
1. To run the MuBCM metamodel:

```bash
cd GraphMM_MuBCM
python run_surrogate_ISK_active.py
python run_surrogate_ICN_active.py
python run_surrogate_VE_active.py
python run_MuBCM_metamodel_active.py
```

2. Results will be saved in the `results/` directory、
3. Check whether the variable is correctly coupled using the plotting functions in the script

```bash
python plot_coupling.py
```



# Citation

If you use GraphMM in your research, please cite our papers: \url{}

# Copyright

© 2024 GraphMM Project Contributors (contact: <a href="mailto:chenxi.wang@salilab.org">chenxi.wang@salilab.org</a>). All rights reserved. This project and its contents are protected under applicable copyright laws. Unauthorized reproduction, distribution, or use of this material without express written permission from the GraphMM Project Contributors is strictly prohibited. For inquiries regarding usage, licensing, or collaboration, please contact the project maintainers.
