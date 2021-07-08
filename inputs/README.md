# Bat-Can Input Files

Bat-Can input files contain all the necessary information to run your battery simulation(s).  At a minimum, they must include the following information: 

- A `cell-description` field, which includes information sufficient to describe the anode, separator, and cathode models.
- A `parameters` field, which includes the experimental conditions (e.g., temperature and pressure), as well as a `simulation` field, which describes the type of experiment to be simulated, and the associated conditions/parameters.
- All necessary inputs to describe the material phases in [Cantera](cantera.org), includes definitions for each `phase`, `species`, and `reaction`.

This can indeed be quite a bit of information, but the intent is to gather it all in a single file that can be saved for posterity, so that results may be easily associated with teh relevant inputs, and reproduced as neeed.

## Sample Inputs

To assist you, the following input files come with `bat-can`:

- `LiMetal_PorousSep_spmLCO_input.yaml`: a dense Li metal anode, porous separator with carbonate electolyte (1M LiPF6 in EC:PC), and SPM LiCOO2 cathode, cycled at constant current.
- `LiMetal_Resistor_spmLCO_input.yaml`: a Li metal anode paired with a LiCOO2 cathode, cycled at constant current. The separator is modeled as a simple ionic resistor.
- `spmGraphite_PorousSep_spmLCO_input.yaml`: a SPM graphite anode, porous separator with carbonate electolyte (1M LiPF6 in EC:PC), and SPM LiCOO2 cathode, cycled at constant current.
- `spmGraphite_Resistor_spmLCO_input.yaml`: a SPM graphite anode paired with a SPM LiCOO2 cathode, cycled at constant current. The separator is modeled as a simple ionic resistor.

Feel free to copy, rename, and edit these files, as desired.

## Input Templates

In addition, we provide templates for various input file components, which you can copy, paste, and edit into a `yaml`-based input file of your own design.  In this way, you can mix-and-match components, as desired.

Templates are stored in the following folders:
- `cantera_templates`: demonstrate how to specify the thermo-kinetic inputs for various phases, in Cantera.
- `electrode_templates`: demonstrate how to specify the necessary parameters for various electrode model types.
- `separator_templates`: demonstrate how to specify the necessary parameters for various electrolyte separator model types.
- `simulation_templates`: demonstrate how to set up  model simulations of various experiment types (e.g. constant-current cycling, cyclic voltammetry, electronic impedance spectroscopy).

As noted in each file, the templates are meant to demsontrate the required fields.  You will need to pay close attention that the field entries are correct.  Notably, you will need to make sure that the phase names provided in the anode, separator, and cathode fields correspond to Cantera phase names defined elsewhere in the file.
