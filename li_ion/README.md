# p2d_li_ion_battery
Pseudo-2D Newman-type model of a Li ion battery

# Table of contents
1. [Building required version of Cantera](#1.-Installation-requirements)
2. [Code structure](#2.-Code-structure)
3. [Code parameters](#3.-Code-parameters)

# 1. Installation requirements
The version of Cantera required in this branch is not on the main Cantera Github repository; therefore, it will be necessary to create a new remote and build the source code from there to run the MHC branch of BatCan

The modified Cantera source code can be found [here](https://github.com/decaluwe/cantera/tree/charge-transfer). Once this repository is added as a remote, create a new local branch and checkout the `charge-transfer` branch. Then, build Cantera from source as outlined in the ReadMe for the BatCan repository. All other dependencies remain the same.

# 2. Code structure

The code structure for `li_ion` in `BatCan` is composed of several modules that handle various tasks for implementing the model. There are five separate modules, which handle: user inputs, model initialization, model running, model functions (transport calculations, interfacing with Cantera, and plotting), and post processing. 

## 2.1 Model inputs - li_ion_battery_p2d_inputs.py
This module takes user inputs for a variety of parameters that go into the model. These include:
- discretization of cell components
- desired C-rate
- kinetics method for each electrode interface
- electrolyte transport model
- initial SOC of the cell
- Cantera related information for creating objects
- geometric and transport parameters for all phases

## 2.2 Model initialization - li_ion_battery_p2d_init.py
The user does not typically need to modify this module. The module imports the `Inputs` class from that module and creates the required Cantera objects, class objects for the cell components, and the initial solution vector to pass to the solver. 

Additionally, this module calculates cell level parameters such as the total cell thickness, the total cell porosity, and total cell density based on the parameters for each component. 

## 2.3 Model runner and residual function - li_ion_battery_p2d_model.py
This module contains the solver initialization and running functionality, as well as the residual function containing the governing equations for the model. __This is the primary runner file for the model.__ Within `main()` several things are done:

- The built-in figure creation is initialized using the `setup_plots()` external function
- The solver is setup for each relevant stage of the model (i.e. equilibration, charging, etc.)
- The dataframes for each stage of the model are post-processed using Pandas and assigned column headers
- The data is exported to `.csv` files for further processing if desired

Below `main()` the residual for the solver (Assimulo) is defined within the class `li_ion(Implicit_Problem)` which is an extension of a builtin problem type for Assimulo. Within this class is the function `res_fun()` where the full residual is defined for the model.

The general process in each node `i` (with exceptions such as in the separator or for a solid lithium anode) goes as follows.

1. The boundary conditions are set for fluxes and currents from the interface with node `i-1`
2. The states of the nodes `i` and `i+1` are read from the solution vector.
3. Chemical production rates are read from Cantera for all interfaces
4. The Faradaic current of node `i` is calculated
5. The boundary conditions are set for fluxes and currents from the interface with node `i+1`
6. The double-layer charging current within the node, `i_dl`, is calculated
7. The residual is calculated for all relevant governing equations at node `i`

The general governing equations are (but not all of which are used in every phase/component):

- Conservation of elements in solid phase (species molar density [kmol/m^3])
- Conservation of elements in electrolyte phase (species molar density [kmol/m^3])
- Conservation of charge in electrode or electrolyte phase (double layer potential)
- Conservation of charge (algebraic constraint)

The parameters used in these governing equations as well as in the transport calculations can be found in __3. Code parameters__ of this ReadMe

## 2.4 Model functions - li_ion_battery_p2d_functions.py
This module contains off-loaded functions that are used within the residual function for the model. These functions are:

- `set_state()` which sets concentrations and electric potentials of the Cantera objects representing the various phases in a given electrode and returns a dictionary of concentrations, electric potentials, and chemical production rates
- `set_state_sep()` which sets the concentrations and electric potentials of the Cantera object representing the electrolyte in the separator and returns a dictionary of concentrations and electric potentials
- `dilute_flux()` which calculates species flux and ionic current based on whichever transport model is chosen (using effective diffusion coefficients)
- `solid_flux()` which calculates diffusion of species within an electrode using an intercalation model
- `setup_plots()` which sets up the built-in plots used when running the model


## 2.5 Model post-processing - li_ion_battery_p2d_post_process.py
This module contains functions to plot and post-process data from the model output. These functions are:

- `plot_potential()` which processes and plots the cell potential 
- `plot_electrode()` which plots the intercalation fraction of relevant electrodes
- `plot_elyte()` which plots concentration of species within the electrolyte phase
- `plot_cap()` which plots voltage vs. capacity of the cell during charge/discharge and calculates energetic and Coulombic efficiencies
- `tag_strings()` which processes the Pandas dataframes to collect the tags used in the Pandas dataframe
- `Label_Columns()` which generates the tags used for column headers in the Pandas dataframe

# 3. Code parameters

The input parameters for geometry and transport are given in the table below

| Code name | Description | Default value | Source |
|-----------|-------------|---------------|--------|
| eps_solid_an | Volume fraction of solid phase in anode | 0.8 [-] | Assumed |
| H_an | Anode thickness | 25e-6 [m] | Assumed |
| C_dl_an | Double-layer capacitance of anode | 1.5e-2 [F/m^3] | Assumed |
| eps_elyte_sep | Volume fraction of electrolyte in separator | 0.5 [-] | Assumed |
| H_elyte | Separator thickness | 25e-6 [m] | Assumed |
| tau_sep | Separator tortuosity | 1.6 [-] | Assumed |
| sigma_sep | Separator ionic conductivity | 50.0 [S/m] | Assumed |
| rho_sep | Separator density assuming HDPE | 970 [kg/m^3] | [ref] |
| eps_solid_ca | Volume fraction of solid phase in cathode | 0.5 [-] | Assumed |
| tau_ca | Cathode tortuosity | 1.6 [-] | Assumed |
| r_p_ca | Average pore radius in cathode | 5e-6 [m] | Assumed |
| d_part_ca | Average particle diameter in cathode | 4e-6 [m] | [ref] |
| overlap_ca | Percentage overlap of cathode particles | 0.4 [-] | Assumed |
| H_ca | Cathode thickness | 50e-6 [m] | Assumed |
| sigma_carbon | Average conductivity of carbon | 125165 [S/m] | [ref] |
| sigma_LFP | Average conductivity of LFP | 2.2e-7 [S/m] | [ref] |
| C_dl_ca | Double-layer capacitance of cathode | 1.5e-2 [F/m^2] | Assumed |
| D_Li_ca | Diffusion coefficient of Li in LFP | 4.23e-16 [m^2/s] | [ref] |
| D_Li_elyte | Diffusion coefficient of electrolyte species (vector) | [1e-12, 1e-12, 1e-10, 3e-11]  [m^2/s] | [ref] |
