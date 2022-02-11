
#%%
import pandas as pd
import os

#https://doi.org/10.1016/j.electacta.2014.09.074 for refdata
refpath = 'D:\projects\Data\ToProcess' #change this to file directory
path2 = os.path.abspath(os.getcwd())+'/outputs'
modelfile = 'output.csv'
comparison = 'Refdata.xlsx'
#%%

def interpolate(voltage1, voltage2, capacity1, capacity2, capacity3):
    voltage3 =  (voltage2 - voltage1)*(capacity3-capacity1)/(capacity2-capacity1) +voltage1
    return voltage3

def locater1(refcapacity, refvoltage):
    lowerneighbour_ind = RefData[RefData['Capacity'] < refcapacity]['Capacity'].idxmax()
    higherneighbour_ind = lowerneighbour_ind + 1
    capacity1 = RefData['Capacity'][lowerneighbour_ind]
    capacity2 = RefData['Capacity'][higherneighbour_ind]
    capacity3 = refcapacity
    voltage1 = RefData['Voltage'][lowerneighbour_ind]
    voltage2 = RefData['Voltage'][higherneighbour_ind]
    v3 = (interpolate(voltage1, voltage2, capacity1, capacity2, capacity3))
    SSR = (refvoltage - v3)**2
    return SSR

def SSRmain(folder_name):
    SumSR = 0
    CodeData = pd.read_csv(path2+"/" + folder_name + "/" + modelfile)
    cyclestart = CodeData[CodeData['cycle'] > 0]['cycle'].idxmin() + 1
    chargestart = CodeData[CodeData['cycle'] > 1]['cycle'].idxmin()
    Voltagesort = CodeData.filter(like='phi_ed')
    Phi_ed_loc = Voltagesort.columns[-1]
    Desiredcolumn = Voltagesort[Phi_ed_loc][cyclestart:chargestart]
    Desiredcolumn = Desiredcolumn.reset_index()
    for x, y in enumerate(CodeData['capacity'][cyclestart:chargestart]):
        refvoltage = Desiredcolumn[Phi_ed_loc][x]
        SumSR += locater1(y, refvoltage)
    return SumSR


def locater2(dataset, refcapacity1, refvoltage1):
    new = CodeData['capacity'][cyclestart:chargestart]
    lowerneighbour_ind = new[new < refcapacity1].idxmax()
    higherneighbour_ind = lowerneighbour_ind + 1
    capacity1 = new[lowerneighbour_ind]
    capacity2 = new[higherneighbour_ind]
    capacity3 = refcapacity
    print(capacity1, capacity2, capacity3)
    voltage1 = Desiredcolumn[Phi_ed_loc][lowerneighbour_ind]
    voltage2 = Desiredcolumn[Phi_ed_loc][higherneighbour_ind]
    v3 = (interpolate(voltage1, voltage2, capacity1, capacity2, capacity3))
    SSR = (refvoltage - v3)**2
    return SSR

def SSRmain3(folder_name, dataset):
    SumSR = 0
    CodeData = pd.read_csv(path2+"/" + folder_name + "/" + "output.csv")
    cyclestart = CodeData[CodeData['cycle'] > 0]['cycle'].idxmin() + 1
    chargestart = CodeData[CodeData['cycle'] > 1]['cycle'].idxmin()
    Voltagesort = CodeData.filter(like='phi_ed')
    Phi_ed_loc = Voltagesort.columns[-1]
    Desiredcolumn = Voltagesort[Phi_ed_loc][cyclestart:chargestart]
    Desiredcolumn = Desiredcolumn.reset_index()
    for x, y in enumerate(RefData[dataset]['Capacity']):
        refvoltage1 = RefData[dataset]['Voltage'][x]
        SumSR += locater2(dataset, y, refvoltage)
    return SumSR




#%%
ID_key = {"CPCN04": "50CP50CNT0.4", "CP04":"CP0.4", "CPCN05":"50CP50CNT0.5", "CP05": "CP0.5", "CPCN06":"50CP50CNT0.6", "CP06":"CP0.6"}

#%%
SSRarray = []
altSSRarray = []
for folder_name in os.listdir(path2):
    if "1045" in folder_name:
        string = folder_name
        array = string.split("_")
        dataset = ID_key[array[0]]
        RefData = pd.read_excel(refpath + "/"+ comparison, sheet_name= dataset)
        SSR_file = SSRmain(folder_name)
        SSRarray.append(SSR_file)

print(SSRarray)
# print(altSSRarray)


#result = [locater("50CP50CNT0.4", x, refcapcity) for x, refcapcity in enumerate(CodeData['Capacity'])]
#print(result)

# print (Data['50CP50CNT0.4']['Capacity'][row])
# lowerneighbour_ind = Data2[Data2['Capacity'] < Temp]['Capacity'].idxmax()
# higherneighbour_ind = Data2[Data2['Capacity'] > Temp]['Capacity'].idxmin()
# v3 = interpolate(Data2['Capacity'][higherneighbour_ind], voltage2,Data2['Capacity'][lowerneighbour_ind], Data2['Capacity'][higherneighbour_ind], capacity3)
# SSR += (Voltage_test[i] - v3)**2

# %%
# Official Playlist
# Small World - Bump of Chicken
# Taxi Driver - Rkomi
# 30 - UVERworld