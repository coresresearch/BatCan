import pandas as pd
import numpy as np
from math import floor
from matplotlib import pyplot as plt
import matplotlib as mpl

tick_font = 10
label_font = 12
legend_font = 8
font = 'Arial'
lw = 2.0 # linewidth
cmap = mpl.colormaps['plasma']
mpl.rcParams['font.family'] = 'Arial'

# Function to format the ticks and labeling
def tick_format(ax):
    #ax.set_ylim((1.8, 2.5))
    ax.tick_params(direction='in',top=True, right=True)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(tick_font)
        tick.label1.set_fontname(font)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(tick_font)
        tick.label1.set_fontname(font)

ncolors = 9 # how many species?
val_colors = np.linspace(0, 1, ncolors)
val_colors2 = np.delete(val_colors.copy(), [2,4]) # 2 less species

labels = [r"$\rm S_8$", r"$\rm Li_2S_8$", r"$\rm Li_2S_7$", r"$\rm Li_2S_6$", r"$\rm Li_2S_5$", r"$\rm Li_2S_4$", r"$\rm Li_2S_3$", r"$\rm Li_2S_2$", r"$\rm Li_2S$"]
labels2 = np.delete(labels.copy(), [2,4])

filename_data = 'outputs/Fitting/0.1C Data.csv'
solution_data = pd.read_csv(filename_data, header=None)
Cap_data = solution_data.iloc[:,0].to_numpy()
V_cell_data = solution_data.iloc[:,1].to_numpy()

filename_Tsuzuki = 'outputs/Li_PorousSep_LiSulfur_Tsuzuki/output_Test_output1.pkl'
solution_Tsuzuki_df = pd.read_pickle(filename_Tsuzuki)
solution_Tsuzuki = solution_Tsuzuki_df.reset_index().to_numpy().T
Cap_Tsuzuki = solution_Tsuzuki[3,:]
V_cell_Tsuzuki = solution_Tsuzuki[84,:]
C_k_Tsuzuki = solution_Tsuzuki[91:,:]

filename_Assary = 'outputs/Li_PorousSep_LiSulfur_Assary/output_Test_output1.pkl'
solution_Assary_df = pd.read_pickle(filename_Assary)
solution_Assary = solution_Assary_df.reset_index().to_numpy().T
Cap_Assary = solution_Assary[3,:]
V_cell_Assary = solution_Assary[72,:]
C_k_Assary = solution_Assary[79:,:]

filename_Kuzmina = 'outputs/Li_PorousSep_LiSulfur_Kuzmina/output_Test_output1.pkl'
solution_Kuzmina_df = pd.read_pickle(filename_Kuzmina)
solution_Kuzmina = solution_Kuzmina_df.reset_index().to_numpy().T
Cap_Kuzmina = solution_Kuzmina[3,:]
V_cell_Kuzmina = solution_Kuzmina[72,:]
C_k_Kuzmina = solution_Kuzmina[79:,:]

fig_volt = plt.figure(figsize = (4, 3),label ='0.1 C')
plt.plot(Cap_data, V_cell_data, 'ko', markersize=1)
plt.plot(Cap_Tsuzuki, V_cell_Tsuzuki, linewidth = lw, color = cmap(1/3),label ='Tsuzuki')
plt.plot(Cap_Assary, V_cell_Assary, linewidth = lw, color = cmap(2/3) ,label ='Assary')
plt.plot(Cap_Kuzmina, V_cell_Kuzmina, linewidth = lw, color = cmap(3/3) ,label ='Kuzmina')
#plt.legend(fontsize = legend_font, loc='center left', ncol=1, bbox_to_anchor=(1, 0.5))
plt.legend(fontsize = legend_font)
plt.ylabel('Cell Potential (V)', fontsize = label_font)
plt.xlabel('Capacity (mAh cm$^{-2}$)', fontsize = label_font)
plt.xticks([0, 500, 1000, 1500])
plt.ylim([1.4,3])
ax  = plt.gca() 
tick_format(ax)
plt.tight_layout()
plt.savefig('Discharge.png', bbox_inches='tight', dpi=450)

fig_T = plt.figure(figsize = (4, 3))
for i in range(len(labels)):
    plt_clr = cmap(val_colors[i])
    plt.plot(Cap_Tsuzuki,C_k_Tsuzuki[i],linewidth = lw,color= plt_clr, label = labels[i])
plt.legend(fontsize = legend_font, loc='center left', ncol=1, bbox_to_anchor=(1, 0.5))
plt.ylabel('Elyte Species Conc. (kmol m$^{-3}$)', fontsize = label_font)
plt.xlabel('Capacity (mAh cm$^{-2}$)', fontsize = label_font)
plt.xticks([0, 500, 1000, 1500])
ax  = plt.gca() 
tick_format(ax)
plt.tight_layout()
plt.savefig('C_k_Tsuzuki.png', bbox_inches='tight', dpi=450)

fig_A = plt.figure(figsize = (4, 3))
for i in range(len(labels2)):
    plt_clr = cmap(val_colors2[i])
    plt.plot(Cap_Assary,C_k_Assary[i],color= plt_clr, label = labels2[i])
plt.legend(fontsize = legend_font, loc='center left', ncol=1, bbox_to_anchor=(1, 0.5))
plt.ylabel('Elyte Species Conc. (kmol m$^{-3}$)', fontsize = label_font)
plt.xlabel('Capacity (mAh cm$^{-2}$)', fontsize = label_font)
plt.xticks([0, 500, 1000, 1500])
ax  = plt.gca() 
tick_format(ax)
plt.tight_layout()
plt.savefig('C_k_Assary.png', bbox_inches='tight', dpi=450)

fig_K = plt.figure(figsize = (4, 3))
for i in range(len(labels2)):
    plt_clr = cmap(val_colors2[i])
    plt.plot(Cap_Kuzmina,C_k_Kuzmina[i],color= plt_clr, label = labels2[i])
plt.legend(fontsize = legend_font, loc='center left', ncol=1, bbox_to_anchor=(1, 0.5))
plt.ylabel('Elyte Species Conc. (kmol m$^{-3}$)', fontsize = label_font)
plt.xlabel('Capacity (mAh cm$^{-2}$)', fontsize = label_font)
plt.xticks([0, 500, 1000, 1500])
ax  = plt.gca() 
tick_format(ax)
plt.tight_layout()
plt.savefig('C_k_Kuzmina.png', bbox_inches='tight', dpi=450)
plt.show()

