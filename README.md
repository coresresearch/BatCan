# BatCan
Battery--Cantera: Modeling fundamental physical chemistry in batteries using Cantera. 

This tool allows you to run battery simulations with easily editable and extensible thermochemistry via [Cantera](cantera.org).

1. [Repository Contents](#repository-contents)
2. [Installation](#installation-instructions)
3. [Running the model](#running-the-model)
4. [Sample results](#sample-results)
5. [Current capabilities](#current-status-of-the-software)


# Repository contents
- `bat_can.py`: this is the main file that runs the code.  In general, the code is run on the command line via `python bat_can.py` (more on this [below](#Running-the-model))
- `bat_can_fit.py`: this is a version of `bat_can.py` that lets you fit against reference data. See [capabilities](#current-status-of-the-software) for more info.
- `bat_can_init.py`: This file reads the user input file and initializes the model.  It is called internally by `bat_can.py`.
- `Simulations` folder. Simluation packages which define different simulation types/routines:
    1. `CC_cycle`: constant current galvanostatc cycling.
    2. `potential_hold`: a series of potentiostatic holds of variable duration.
    3. `galvanostatic_hold`: a series of galvanostatic holds of variable duration.
    4. `cyclic_voltammetry`: cyclic voltammetry experiment.
- Electrode model packages:
    1. `single_particle_electrode`: the standard "single particle model" approach to a porous electrode, including radial discretization and transport.
    2. `dense_electrode`: Model for a dense, thin-film electrode.  Currently demonstrated for a lithium metal anode, but could be used for other purposes.
    3. `conversion_electrode`: Electrode that converts a solid reactant phase into a separate solid product phase. Currently demonstrated for a sulfur cathode.
    4. `air_electrode`: metal-air electrode model, which interfaces to a gas flow channel. Currently demonstrated for a lithium-oxygen cathode.
- Electrolyte model packages:
    1. `ionic_resistor`: Simple ionic resistor with no chemical composition dynamics.
    2. `porous_separator`: porous inert separator filled with liquid electrolyte.
- Submodels: functions and routines that are used by multiple parts of the code.
    1. `transport`: functions to describe transport phenomena. Currently includes a dilute solution electrolyte transport model and a radial solid diffusion model.
- `inputs`: folder with all input files.
- `outputs`: folder where all model outputs are saved.
- `derivation_verification`: Notes and documents to describe model development, governing equations, etc. (currently under development)
# Installation Instructions

In order to use the BatCan suite, it is necessary to download and install:
- [Cantera](cantera.org)
- [Numpy](numpy.org)
- [Scikits.odes](https://pypi.org/project/scikits.odes)
- [Ruamel.yaml](https://pypi.org/project/ruamel.yaml/)
- [Matplotlib](matplotlib.org)
- [Pandas](https://pandas.pydata.org/)

These can all be installed an managed via [Anaconda](anaconda.org)

For example, to create a conda environment `bat_can` from which to run this tool, enter the following on a command line, terminal, or Anaconda prompt:
```
conda create --name bat_can --channel conda-forge cantera scikits.odes matplotlib numpy ruamel.yaml pandas
```
You can replace `bat_can` with whatever name you would like to give this environment. After this completes, activate the environment:
```
conda activate bat_can
```
(again, replacing `bat_can`, as necessary, if you've named the environment something different). When you're done using the tool and want to switch back to your base software environment, run:
```
conda deactivate
```

# Running the Model 
To run the model, there are two main steps:
1. [Choose or develop an input file](#Input)
2. [Run the model](#Run-the-Model)

## Input 
The input file provides all the necessary information to `bat_can` program so that it may run your simulation.

The input file includes three primary sections:
- A description of the battery components (anode, electrolyte separator, and cathode), including model type for each, geometry and microstructural parameters.
- A description of the simulation to run and parameters to specify the necessary operating conditions.
- A Cantera input section, used to create objects that represent the phases of matter present, the interfaces between then, and the thermodynamic, chemical kinetic, and transport processes involved.

If you would like to create your own input, there are template folders which demonstrate how to specify inputs for the various electrode, separator, and simulation types, as well as a number of relevant Cantera phases.  You can copy and paste these snippets into a single input file to customize.  There are also a large number of working input files located in the `inputs` folder, which are meant to demonstrate and test the minimum functionality of the code.  Locate one that you would like to use, modify an existing file to suit your purposes, or copy a file, save it to a new name, and edit as necessary.

At present, the input file must be saved to the `inputs` folder.

## Run the Model
The model is run from a command line or terminal by invoking python, the `bat_can.py` name, and providing the name of your input file (with or wihout the `.yaml` suffix) by assigning the keyword `--input`. For example, if your input file is located at `inputs/my_input.yaml`, you would run:
```
python bat_can.py --input=my_input
```
Again, including the file extension is optional.  The command:
```
python bat_can.py --input=my_input.yaml
```
would also work.
# Sample results.
Below is an example of the model output, for a Li metal anode, porous separator with liquid carbonate electrolyte, and single-particle model of an LCO cathode, cycled 5 times at a rate of 0.01C (Note that the kinetic and transport parameters have not been tuned or even sourced from literature; this is for demonstration purposes only 🙂 ).

![Sample output image](sample_output.png)

# Current status of the software 
(as of 31 January, 2021)

This software is currently in the development phase. 

## Models
Models are configured for either galvanostatic or potentiostatic operation. 
- The `ionic_resistor` and `porous_separator` separator models are complete. The `porous_separator` model currently only allows dilute solution approximation transport calcualtions, via the Poisson-Nernst-Planck equation.
- The `dense_electrode` model is complete 
- The `single_particle_model` electrode is complete.
- Simulation models (`CC_cycle`, `potential_hold`, and `cyclic_voltammetry`) are all complete, but there are always new and more flexible user inputs that can be enabled 🙂.

## Simulation speed.
We have added two new features to improve the software speed:
- The software automatically calculates the width of the Jacobian band, and passes this information to the solver, which save significant computational time (roughly an order of magnitude faster).
- The software now allows multiprocessing. This only helps if your input file runs multiple simulations, though (for example, cycling the same battery over a range of C-rates.) Use the `cores` flag on your model input.  For example, to run the simulations describe in a file `my_input.yaml` in parallel on 3 cores, you would type:
```
python bat_can.py --input=my_input.yaml --cores=3
```

## Data Fitting
If you want to try and fit your model to experimental data, there are now capabilities to do this by running `bat_can_fit.py` instead of `bat_can.py`.  This is still in early stages, and only available for a few types of fitting parameters.

Fitting parameters and their values are specified in a new `fit-parameters` field in the input file.  See `inputs/LiO2_Fitting_example.yaml` for an example.

# Making changes
Adding new features is relatively easy, so please click on `Issues` above and create a new issue if there is something you would like to see! 

If you would like to help contribute to the software, please do! If you are uncertain of what to do, or have an idea and want to run it by us, maybe create an issue on the issues page, where we can discuss.  Or else, feel free to fork a copy of this repo, make changes, and make a pull request.
