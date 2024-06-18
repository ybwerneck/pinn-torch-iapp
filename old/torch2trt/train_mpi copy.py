from mpi4py import MPI
from train_protocol import train
from PinnTorch.dependencies import *
import os

import itertools
size= MPI.COMM_WORLD.Get_size() 
rank = MPI.COMM_WORLD.Get_rank()
print(rank)
print(size)
import pickle


import os

dir="./batch_results/"
import sys

dir=sys.argv[1] 

# Define the list
hl = [(nn.ReLU, 128), (nn.SiLU, 128), (nn.Tanh, 128), (nn.ELU, 128)]

# Define the input array with sizes
sizes = [2, 3, 4]

# Initialize the final list to store sets of combinations for each size

simulations=[]
# Generate combinations for each size and add to the final list
for size in sizes:
    unique_combinations = set(itertools.combinations_with_replacement(hl, size))
    for u in unique_combinations:
    
        simulations.append(
            {
                "model_params":{
                    "hidden_layers":u
                }
            }
            
        )


if(True):
    # Number of processes per node (ppn)


    # Total number of GPUs available
    total_gpus = torch.cuda.device_count()

    # Number of GPUs each process should use
    gpus_process = rank%int(sys.argv[2])

    # Log the information
    if rank == 0:
        print(f"Total GPUs available: {total_gpus}") 
        
    print(f"Process {rank} is using GPU {gpus_process}")
  
    for i,_ in enumerate(simulations):
        
      if(i%size==rank):
        pdir=dir+f"simulation{i}"
        os.mkdir(pdir)
        with open(pdir+'/my_dict.pkl', 'wb') as f:
            pickle.dump(simulations[rank], f)

        
        
        train(simulations[rank]["model_params"],outputfolder=pdir,gpuid=gpus_process)
