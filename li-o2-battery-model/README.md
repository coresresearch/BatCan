# lio2-battery-model
1 Dimensional Li-O2 Battery model

This 1-D model simulates lithium-O2 battery operation.

The simulation domain includes:

- Dense Li metal anode (to do)
- Porous electrolyte separator filled with liquid carbonate electrolyte (to do)
- Porous carbon cathode, with pores filled by electrolyte and Li2O2 discharge product. 

Physio-electro-chemical processes modeled include:

- Charge transfer at the Li anode/electrolyte interface (to do)
- Electrochemical diffusion through the porous electrolyte separator, with either dilute or concentrated solution theory (to do).
- Electrochemical diffusion through the electrolyte phase of the porous cathode (to do).
- Charge transfer within the porous cathode, at the electrolyte/carbon host interface.

## Current status:
The model currently implements a zero-dimensional (aka 'single particle model') model of just the cathode, where charge neutrality dictates that the ionic current into the separator equals the external current into the cathode.

Updates in the near future will include:

- Discretize the cahode along one dimension
- Add electrolyte separator and li anode
- Discretize the electrolyte separator

## Installation: 

The model requires several dependencies, including:

- Cantera v2.5.0 or greater
- NumPy
- Matplotlib
- SciPy

It is highly recommended that you install and manage dependencies via Conda.  Download and instal Miniconda (the full Anaconda package is okay, but not required.)

Open up a terminal or anaconda prompt window and install the development version of Cantera.  The following command will install it into an environment named `li-o2-battery`.  You can replace this with any environment name you desire.

```
conda create -name li-o2-battery --channel cantera/label/dev cantera
```

Respond with `y`, when prompted.  Then activate this environment:

```
conda activate li-o2-battery
```
and install the other required dependencies:

```
conda install numpy matplotlib scipy
```

When you are done using this software environment, you can return to your normal "base" environment:

```
conda deactivate
```

Next, download or clone the files in this repository, either manually as a .zip file, or using git for version control:

```
git clone https://github.com/coresresearch/lio2-battery-model.git
```
## Running the Model:
1. Edit the input files `li_o2_inputs.py` and `lithium_o2_battery.yaml` (these will eventually be merged into a single input file). You can vary the thermo-chemical properties, battery geometry or microstructure, and/or charge discharge parameters.
2. Either via the command line, or using a Python IDE (e.g. Atom, VS Code, or PyCharm), run the file `li_o2_model.py`

