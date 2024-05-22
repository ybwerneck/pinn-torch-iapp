from mpi4py import MPI
from train_protocol import train
from PinnTorch.dependencies import *
import os


import itertools
SA= MPI.COMM_WORLD.Get_size() 
rank = MPI.COMM_WORLD.Get_rank()
print(rank)
print(SA)
import pickle

log_file = f"logs/mpi_log{rank}.txt"

import os

dir="./batch_results/"
# Define the list
hl = [(nn.SiLU, 32),(nn.SiLU, 64),(nn.ELU, 32),(nn.ELU, 64)]

# Define the input array with sizes
sizes = [2]

# Initialize the final list to store sets of combinations for each size
simulations = []

# Function to create a sorting key
def sort_key(array_of_tuples):
    # Sort each tuple by (num, cls.__name__)
    return [array_of_tuples[i][0].__name__ for i in range (len(array_of_tuples))]
# Generate combinations for each size and add to the final list
result=[]
for size in sizes:
    result=[]
    unique_combinations = set(itertools.combinations_with_replacement(hl, size))
    for combination in unique_combinations:
        permutations = itertools.product(combination, repeat=size)
        result.extend(permutations)
    result=set(result)
    sorted_combinations = sorted(result, key=sort_key)  # Ensure the combinations are ordered consistently
    for u in sorted_combinations:
        simulations.append(
            {
                "model_params": {
                    "hidden_layers": u
                }
            }
        )

# Example: Print the generated combinations for verification
for idx, comb in enumerate(simulations):
    print(f"Simulation {idx+1}: {comb}")
    
if(True):
        # Number of processes per node (ppn)


        # Total number of GPUs available
        total_gpus = torch.cuda.device_count()

        # Number of GPUs each process should use
        gpus_process = rank%2
        # Log the information
        with open(log_file, "w") as log:
            log.write(f"Total GPUs available: {total_gpus} \n") 
            
            log.write(f"Process {rank} is using GPU {gpus_process} \n")
            # Check if CUDA is available
            if torch.cuda.is_available():
                # Print all available GPU devices
                for i in range(torch.cuda.device_count()):
                    log.write(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                log.write("No GPU available. Only CPU is available.")
    
    
        for o,_ in enumerate(simulations):
            pdir=dir+f"simulation{o}"

            if(o%SA==rank):
                os.mkdir(pdir)
                with open(pdir+'/my_dict.pkl', 'wb') as f:
                        pickle.dump(simulations[o], f)
                with open(log_file, "a") as log:
                    log.write(f"\n RANK{rank}, runnin simu {o} params:\n {simulations[o]["model_params"]} \n in dir {pdir}")
                train(simulations[o]["model_params"],outputfolder=pdir,gpuid=gpus_process)