from PinnTorch.dependencies import *
from PinnTorch.Net import *
from PinnTorch.Trainer import *
from PinnTorch.Validator import *
from PinnTorch.Loss import *
from PinnTorch.Loss_PINN import *
from PinnTorch.Utils import *
from custom_Validator import gen_recursive_validator
def ensure_at_least_one_column(x):
    # If x is 1-dimensional, reshape to (len(x), 1)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    # If x is already 2-dimensional, return it unchanged
    elif x.ndim == 2:
        return x
    else:
        raise ValueError("Input array must be 1D or 2D")
    

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val
# Check if GPU is availabls
def train(model_params,outputfolder,gpuid):
        device = torch.device(f"cuda:{gpuid}")##torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        tf=20
        ##Model Parameters
        k_range = (0,1)
        v_range = (0,1)
        u_range = (0,1)
        t_range=(0,tf)
        nP=2

        ##Model Arch
        input_shape = 3 
        output_shape = nP
           
        hidden_layer = model_params["hidden_layers"]
        dtype=torch.float32
        model = FullyConnectedNetworkMod(input_shape, output_shape, hidden_layer,dtype=dtype).to(device)
        trainer=Trainer(model,output_folder=outputfolder,print_steps=100,val_steps=1000)

        print(model)

        ICs=[0.5,0]  

        data_int,data_outt=LoadDataSet("training_data/treino/",data_in=["U.npy","V.npy","K.npy"],device=device,dtype=dtype)
        
        
        data_inv,data_ouv=LoadDataSet("training_data/validation/",data_in=["U.npy","V.npy","K.npy"],device=device,dtype=dtype)
        time_series_data=LoadDataSet("training_data/validation_ts/",data_in=["T.npy","U.npy","V.npy","K.npy"],device=device,dtype=dtype)


       


        print(np.shape(data_outt))
        trainer.add_loss(LPthLoss(data_int,data_outt,1024,2,True,device,"Data Loss"))

        ##Validator
        trainer.add_validator(Validator(data_inv,data_ouv,"val",device,dump_f=1 ,dump_func=gen_recursive_validator(time_series_data,10)))


        trainer.train(1000000)





        print(model)

