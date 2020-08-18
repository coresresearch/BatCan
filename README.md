# BatCan
Battery--Cantera: Modeling fundamental physical chemistry in batteries using Cantera. 

# Repository contents
- `BatCan/functions` contains external functions that are used by more than one model in `BatCan`. Currently this is for concentrated solution theory transport calculations
- `BatCan/li_ion` contains a Pseudo-2D lithium-ion model
- `BatCan/li_o2` contains a 1D lithium-air model

Check the ReadMe of relevant subdirectories for model specific instructions.

# General Installation Instructions

In order to use the BatCan suite, it is necessary to download Cantera. `Batcan/li_ion` requires the development version of Cantera. Instructions for creating an environment with the necessary packages and installing can be found [here](https://cantera.org/compiling/installation-reqs.html#sec-installation-reqs). 

__An important note when creating the environment:__ The solver used for `BatCan/li_ion` has specific package version dependencies, therefore __build an environment with `Python=3.5`__

Once Cantera is built and installed, there are a few more packages required in the working environment. 

For running `BatCan/li_ion`, the packages required are:

- Assimulo (run the following in Anaconda Prompt `conda install -c https://conda.binstar.org/chria assimulo`)
- Pandas
- If using Spyder as an IDE, the version known to work with other packages is `Spyder=3.2.8`.
- Pywin32 might also be required
