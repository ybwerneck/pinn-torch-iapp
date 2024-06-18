from PinnTorch.dependencies import *
from PinnTorch.Net import *
from PinnTorch.Trainer import *
from PinnTorch.Validator import *
from PinnTorch.Loss import *
from PinnTorch.Loss_PINN import *
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
        trainer=Trainer(model,output_folder=outputfolder,print_steps=100,val_steps=1000)

        print(model)

        ICs=[0.5,0]



        def ode(t, x):
                dxdt = [0.1*x[i] for i in range(nP)]
                return dxdt


        def ode(t, x):
            nP = len(x)
            u_i=x[0]
            v_i=x[1]
            dxdt = [10*((1)*(u_i*(u_i-0.4)*(1-u_i))-v_i   ),
                    
                    ((u_i*0.04-0.16*v_i))
                    
                    ]
            return dxdt

       

        def FHN_LOSS(data_in, model):
               
            ts = data_in
            # run the collocation points through the network
            u = model(ts)
            ode_target=ode(ts,u.T)
        
            odeA=0
            # get the gradient
           # print(np.shape(u))


            for i in range(nP):
                ui= u.T[i].view(-1,1)

                tgt=ode_target[i].view(-1,1)


                #Mode1
                dT = grad(ui, ts)[0]          
                odeA += (dT - tgt)**2
            

            # MSE of ODE

            return torch.mean(odeA)






        batch_gen=lambda xs,device:ensure_at_least_one_column(default_batch_generator(xs,[[0,tf]],device))

       #trainer.add_loss(LOSS_PINN(FHN_LOSS,batch_gen,batch_size=4*64),weigth=1)



        

        ##BoundaryLossss

        def f(data_in, model):
    
                u = model(data_in) 

                u0,u1= u.T[0].view(-1,1) , u.T[1].view(-1,1)

                

                t0 = (u0-ICs[0])**2 + (u1-ICs[1])**2



                return torch.mean(t0)
        

        
        batch_gen=lambda size,de:ensure_at_least_one_column(torch.zeros(size,requires_grad=True).to(de).T)
        trainer.add_loss(LOSS_PINN(f,batch_gen,batch_size=2,device=device),weigth=1)

        trainer.add_loss(FHN_LOSS_fromODE(ode,[0,tf],ICs,batch_size=4*64,num_points=2*102400,device=device,dtype=dtype))

        ##Validator
        trainer.add_validator(FHN_VAL_fromODE(ode,[0,tf],ICs,1024,device=device,name="val",dtype=dtype,dump_factor=20))


        trainer.train(10000000)





        print(model)

