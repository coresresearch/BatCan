# Data folder

This folder is meant to hold validation data used for model fitting.  If you want to compare your simulation to data from the literature, place the reference data in this folder (I would recommend making a subfolder, e.g. `data/my-data/discharge.xls`).  The routines are currently set up to read data from excel, but can be easily modified (see `bat_can_fit.py`, line 239), to accomodate other data types.

When you run `bat_can_fit.py`, the simualtion will automatically compare the simulated data to your reference data, plot them together, and calculate the sum of squared residuals $SSR$:
$$ SSR = \sum_i \left(V_{i,\.{\rm ref}} - V_{i,\.{\rm sim}})^2$$
where $V_{i,\.{\rm ref}}$ is the voltage from the reference data, $V_{i,\.{\rm sim}} is the simulated voltage at the same capacity, and the sum is over all $i$ data points from the reference data.