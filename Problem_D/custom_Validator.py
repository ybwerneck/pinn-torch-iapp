
import subprocess
import torch
import numpy as np


def recursive_validator(val_obj,time_series_data,file="Results_plotter.py",factor=1):
    its=10
    data_in_ts,target_ts=time_series_data
    model=val_obj.model
    print(np.shape(data_in_ts))
    xi=val_obj.data_in.to(val_obj.device)
    data_out=torch.zeros((2,len(xi),its + 1),device=val_obj.device)
    print(np.shape(target_ts))
    k=0


    for i in range(its):
        
        xi=model(xi)      

        if(i%1==0):
            data_out[:,:,k]=xi[:,:-1].T
            k+=1
        
        xi=torch.stack((xi.T[0],xi.T[1],val_obj.data_in.T[2].to(val_obj.device)),axis=1)
    
    val_obj.dump_f_def()
    val_obj.dump_f_def(target_ts,data_out.reshape(np.shape(target_ts)),data_in_ts,sufix="_ts")
    subprocess.Popen(f"python {file} {val_obj.folder}/", shell=True, stdout=subprocess.PIPE).stdout.read()



def gen_recursive_validator(time_series_data,factor):
    return lambda val_obj:recursive_validator(val_obj,time_series_data,file="Results_plotter.py",factor=factor)