# wrapper.py
#%%
path = os.path.abspath(os.getcwd())+'/outputs'
import sys
import os
import pandas as pd
import numpy as np
from bat_can import bat_can
from SSRcode import SSRmain
from scipy import optimize as spo
import yaml

#%%
inputfile = ["LiMetal_PorousSep_Air.yaml"]
#Summary_array = [0 for _ in inputfile] forgot what this is for.  It might be useful some day
#%%
ID_key = {"CPCN04": "50CP50CNT0.4", "CP04":"CP0.4", "CPCN05":"50CP50CNT0.5", "CP05": "CP0.5", "CPCN06":"50CP50CNT0.6", "CP06":"CP0.6"}
comparison = 'Refdata.xlsx'
refpath = 'D:\projects\Data\ToProcess' #change this to file directory
test_yaml = 'D:\projects\BatCan\inputs\LiMetal_PorousSep_Air_test.yaml'

#%%

def batcanrun(rate_constant):
    with open(test_yaml, 'r') as file:
        imput_file = yaml.safe_load(file)
    imput_file['cathode-surf-reactions'][0]['rate-constant']['A'] = rate_constant
    with open(test_yaml, "w") as f:
        yaml.dump(imput_file, f)
    foldarray =os.listdir(path)
    bat_can('LiMetal_PorousSep_Air_test.yaml')
    foldarray2 = os.listdir(path)
    folder_name = list(set(foldarray).symmetric_difference(set(foldarray2)))
    SSRarray = []
    for i in folder_name:
        array = i.split("_")
        dataset = ID_key[array[0]]
        RefData = pd.read_excel(refpath + "/"+ comparison, sheet_name= dataset)
        SSR_file = SSRmain(i)
        SSRarray.append(SSR_file)
        print(SSRarray)
    return(sum(SSRarray))

#%%
test_yaml = 'D:\projects\BatCan\inputs\LiMetal_PorousSep_Air_test.yaml'
Done = batcanrun(5629000.0)
print(Done)

#%%
x_start = 5629000.0
result = spo.minimize(batcanrun, x_start)
print("x:", result.x)
print("SSR:", result.fun)
#%%