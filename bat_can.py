# bat_can.py
"""
    BatCan - Battery Modeling in Cantera

    This file reads in the user input, runs the simulation, and then produces any requested output (saved data file, preliminary plots, etc.)

"""
# Import modules
import importlib # allows us to import from user input string.
from bat_can_init import initialize as init

def bat_can(input = None):
    if input is None:
        raise ValueError('Please specify an input file.')
    



# Run from command line:
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()
    
    bat_can(args.input)