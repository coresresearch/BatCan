from bat_can import bat_can
import numpy as np
#from multiprocessing.pool import ThreadPool as Pool
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt
import matplotlib
import time
owd = os.getcwd()
os.chdir(owd)

def fitting(input):
    ref_data = 'outputs/Fitting' #/Li_PorousSep_Sulfur_Assary_2step'
    data_folder = 'outputs/Fitting/' + input
    output_folder = "outputs/Fitting/" + input + "_"

    lw = 2.0
    grad = np.linspace(0, 1, 7)
    species_cmap = np.zeros((len(grad),4))
    for i, val in enumerate(grad):
        species_cmap[i] = matplotlib.cm.plasma(val)

    def get_file_name(find_string, dir_path, **kwargs):
        flag = kwargs.get('flag', None)
        full_list = os.listdir(dir_path)
        if flag:
            file_name = [file for file in full_list if find_string in file]
        else:
            file_name = [file for file in full_list if find_string in file][0]

        return file_name

    def df_name(file_name_list, file_number):
        file_name = file_name_list[file_number]
        df_name = file_name.split('_')
        return df_name[0]

    def read_file(file_name_list):
        df_dict = {}
        for i in np.arange(0, len(file_name_list)):
            name = df_name(file_name_list, i)
            df_dict[name] = pd.read_csv(file_name_list[i])
        return df_dict

    def list_files(dir):
        data = {}
        for root, dirs, files in os.walk(dir):
            for i, dir in enumerate(dirs):
                folder = root + "/" + dir
                file_name = get_file_name('.csv', folder, flag=1)
                os.chdir(owd)
                os.chdir(folder)
                data['set'+str(i+1)] = read_file(file_name)
                os.chdir(owd)
                n_sets = i+1
                print('Importing data set', n_sets, 'of', len(dirs))

        return data, n_sets

    def format_fig(n_fig):
        fs = 14
        font = 'Times New Roman'

        fig = plt.figure(n_fig)
        ax = fig.add_axes([0.2, 0.2, 0.6, 0.75])
        ax.tick_params(direction='in')
        fig.set_size_inches((6., 4.))

        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fs)
            tick.label1.set_fontname(font)
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fs)
            tick.label1.set_fontname(font)

    def add_zeros():

        return

    "Import reference data"
    ref_data_file = get_file_name('.csv', ref_data, flag=1)
    model_data_file = get_file_name('.csv', data_folder, flag=1)

    print('importing reference data')
    os.chdir(owd)
    os.chdir(ref_data)
    ref_data = read_file(ref_data_file)
    os.chdir(owd)

    print('importing model data')
    os.chdir(owd)
    os.chdir(data_folder)
    model_data = read_file(model_data_file)
    os.chdir(owd)

    Cap_model = model_data['output']['capacity'].to_numpy()
    Vcell_model = model_data['output']['phi_ed.1'].to_numpy()

    "Sort reference data into capacity and voltage"
    Cap_ref = ref_data['0.1C Data.csv'].iloc[:,0].to_numpy()
    V_cell_ref = ref_data['0.1C Data.csv'].iloc[:,1].to_numpy()
    #Cap_ref = ref_data['output']['capacity'].to_numpy()
    #V_cell_ref = ref_data['output']['phi_ed.1'].to_numpy()
    Cap_ref = Cap_ref[:] #[::190]
    V_cell_ref = V_cell_ref[:] #[::190]


    "Calculate average spacing of capacity on reference data"
    Cap_ref_avg = np.average(np.ediff1d(Cap_ref))

    "Add zeros to either reference data or the datasets, whichever is shorter"
    if Cap_model[-1] > Cap_ref[-1]:
        Cap_append = np.arange(Cap_ref[-1]*1.001, Cap_model[-1], Cap_ref_avg)
        Cap_append = np.append(Cap_append, Cap_model[-1])
        V_append = np.zeros_like(Cap_append)

        Cap_ref = np.append(Cap_ref, Cap_append)
        V_cell_ref = np.append(V_cell_ref, V_append)
    elif Cap_model[-1] < Cap_ref[-1]:
        Cap_append = np.arange(Cap_model[-1]*1.001, Cap_ref[-1], Cap_ref_avg)
        Cap_append = np.append(Cap_append, Cap_ref[-1])
        V_append = np.zeros_like(Cap_append)

        Cap_model = np.append(Cap_model, Cap_append)
        Vcell_model = np.append(Vcell_model, V_append)

    "Figure 1 plots all data before processing to calculate goodness of fit"
    #format_fig(1)
    #for i, key in enumerate(Cap_dict.keys()):
    #    plt.plot(Cap_ref_dict[key], V_cell_ref_dict[key], linewidth=lw, color=species_cmap[i])
    #    plt.plot(Cap_dict[key], Vcell_dict[key], linewidth=lw, color=species_cmap[i])
    #plt.ylim((1.8, 2.4))

    "Interpolate voltage data to match reference capacity values in order to"
    "   calculate goodness of fit parameters such as SSR"
    Vcell_int = np.interp(Cap_ref, Cap_model, Vcell_model)

    #format_fig(2)
    #for i, key in enumerate(Cap_dict.keys()):
    #    plt.plot(Cap_ref_dict[key], V_cell_ref_dict[key], linewidth=lw, color=species_cmap[0], marker='o')
    #    plt.plot(Cap_ref_dict[key], Vcell_int[key], linewidth=lw, color=species_cmap[i], marker='o')
    #plt.ylim((1.8, 2.4))

    "Calculate SSR for all data sets and print values"
    SSR = 0
    for i, ref_cap in enumerate(Cap_ref):
        if i == 0:
            Cap_prev = ref_cap
            SSR += (V_cell_ref[i] - Vcell_int[i])**2
        else:
            d_cap = ref_cap - Cap_prev
            Cap_prev = ref_cap
            if d_cap >= Cap_ref_avg:
                SSR += (V_cell_ref[i] - Vcell_int[i])**2
    #SSR = sum((V_cell_ref - Vcell_int)**2)

    return SSR

if __name__ == '__main__':
    import argparse

    # Currently, the only command line keyword enabled is --input, to specify
    # the input file location:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    args = parser.parse_args()

    fitting(args.input)
