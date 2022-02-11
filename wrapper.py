# wrapper.py
#%%
path = os.path.abspath(os.getcwd())+'/outputs'
import sys
import os
import pandas as pd
from bat_can import bat_can
from SSRcode import SSRmain

#%%


foldarray =os.listdir(path)
inputfile = "LiMetal_PorousSep_Air"
bat_can(inputfile)
foldarray2 = os.listdir(path)

#%%
ID_key = {"CPCN04": "50CP50CNT0.4", "CP04":"CP0.4", "CPCN05":"50CP50CNT0.5", "CP05": "CP0.5", "CPCN06":"50CP50CNT0.6", "CP06":"CP0.6"}
comparison = 'Refdata.xlsx'
SSRarray = []
refpath = 'D:\projects\Data\ToProcess' #change this to file directory
folder_name = list(set(foldarray).symmetric_difference(set(foldarray2)))[0]
array = folder.split("_")
dataset = ID_key[array[0]]
RefData = pd.read_excel(refpath + "/"+ comparison, sheet_name= dataset)
SSR_file = SSRmain(folder_name)
SSRarray.append(SSR_file)
print(SSRarray)



#%%
# SSRarray = []
# path = os.path.abspath(os.getcwd())+'/outputs'
# refpath = 'D:\projects\Data\ToProcess' #change this to file directory
# for folder_name in os.listdir(path):
#     if "1045" in folder_name:
#         string = folder_name
#         array = string.split("_")
#         dataset = ID_key[array[0]]
#         RefData = pd.read_excel(refpath + "/"+ comparison, sheet_name= dataset)
#         SSR_file = SSRmain(folder_name)
#         SSRarray.append(SSR_file)


#%%
