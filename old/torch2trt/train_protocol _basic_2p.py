from PinnTorch.dependencies import *
from PinnTorch.Net import *
from PinnTorch.Trainer import *
from PinnTorch.Validator import *
from PinnTorch.Loss import *
from PinnTorch.Loss_PINN import *

import subprocess
from PinnTorch.Utils import *
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
        input_shape = 1  
        output_shape = nP
           
        hidden_layer = model_params["hidden_layers"]
        dtype=torch.float32
        model = FullyConnectedNetworkMod(input_shape, output_shape, hidden_layer,dtype=dtype).to(device)
        trainer=Trainer(model,output_folder=outputfolder,print_steps=100,val_steps=10000)

        print(model)



        def ode(t, x):
                dxdt = [0.1*x[i] for i in range(nP)]
                return dxdt

        

        def FHN_LOSS(data_in, model):
               
            ts = torch.linspace(0, tf, steps=10000,device=device).view(-1,1).requires_grad_(True)
            # run the collocation points through the network
            u = model(ts)

            # get the gradient
           # print(np.shape(u))
            u0,u1= u.T[0].view(-1,1) , u.T[1].view(-1,1)

            #ode1
            dT = grad(u0, ts)[0]          
            ode1 = (dT - (0.1*u0))**2
            
            #ode 2
            
            dT = grad(u1, ts)[0]          
            ode2 = (dT - (0.1*u1))**2

            # MSE of ODE

            return torch.mean(ode1+ode2)






        batch_gen=lambda xs,device:ensure_at_least_one_column(default_batch_generator(xs,[[0,tf]],device))

        trainer.add_loss(LOSS_PINN(FHN_LOSS,batch_gen,batch_size=32),weigth=1)
        IC=0.1
        ICs=[IC*(1+i) for i in range(nP)]
        print(ICs)
        ##BoundaryLossss

        def f(data_in, model):
    
                u = model(data_in) 

                u0,u1= u.T[0].view(-1,1) , u.T[1].view(-1,1)

                

                t0 = (u0-ICs[0])**2 + (u1-ICs[1])**2



                return torch.mean(t0)
        

        
        batch_gen=lambda size,de:ensure_at_least_one_column(torch.zeros(size,requires_grad=True).to(de).T)
        trainer.add_loss(LOSS_PINN(f,batch_gen,batch_size=1024,device=device),weigth=1)


        ##Validator
        trainer.add_validator(FHN_VAL_fromODE(ode,[0,tf],ICs,1000,device=device,name="val",dtype=dtype))
        #trainer.add_loss(FHN_LOSS_fromODE(ode,[0,tf],[IC],1000,device=device,name="val",dtype=dtype))

        trainer.train(100000)


        subprocess.Popen(f"python Results_plotter.py {outputfolder}", shell=True, stdout=subprocess.PIPE).stdout.read()



        print(model)

