from PinnTorch.dependencies import *
from PinnTorch.Net import *
from PinnTorch.Trainer import *
from PinnTorch.Validator import *
from PinnTorch.Loss import *
from PinnTorch.Loss_PINN import *

import subprocess
from PinnTorch.Utils import *



# Check if GPU is availabls
def train(model_params,outputfolder,gpuid):
        device = torch.device(f"cuda:{gpuid}")##torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)


        ##Model Parameters
        k_range = (0., 1)
        v_range = (0., 0.12)
        u_range = (-.1, 0.81)
        t_range=(0,20)


        ##Model Arch
        input_shape = 4  
        output_shape = 2   
        hidden_layer = model_params["hidden_layers"]
        dtype=torch.float32
        model = FullyConnectedNetworkMod(input_shape, output_shape, hidden_layer,dtype=dtype).to(device)
        trainer=Trainer(model,output_folder=outputfolder,val_steps=10000)

        print(model)




        ##DataLoss
        data_folder="training_data/treino/"



        #trainer.add_loss(FHN_loos_fromDataSet(data_folder,1024,device=device,loss_type="L2",dtype=dtype,shuffle=True),weigth=1)


        #trainer.add_loss(FHN_loos_fromDataSet("training_data/treinocr/",1024*10,device=device,loss_type="L2",shuffle="True"),weigth=1)
        trainer.add_loss(FHN_loos_fromDataSet("training_data/treinocr2/",4*1024,device=device,loss_type="L2",shuffle=True,dtype=dtype),weigth=2)

        #trainer.add_loss(FHN_loos_fromDataSet("training_data/treinor/",10240,device=device,loss_type="L4",shuffle=True),weigth=1)
        #trainer.add_loss(FHN_loos_fromDataSet("training_data/treinor/",1024*10,device=device,loss_type="L2"),weigth=1)


        ##LOSS_PINN
        def get_derivative( y, x, n):
                # General formula to compute the n-th order derivative of y = f(x) with respect to x
                if n == 0:
                 return y
                else:
                 dy_dx = torch.autograd.grad(y, x, torch.ones_like(y).to(device), create_graph=True, retain_graph=True, allow_unused=True)[0]
                return get_derivative(dy_dx, x, n - 1)

        def FHN_LOSS(data_in, model):
                x,w = model(data_in).T 
                t,u,v,k=data_in.T
                dx_dt=get_derivative(x,data_in,1)[:,0]
                pdeu =  100*(( 1)*(u*(u-0.4)*(1-u))-v +k*0.04 + 0.08)- dx_dt
                pdew =  100*(( 1)*(u*(u-0.4)*(1-u))-v +k*0.04 + 0.08)- dx_dt

                return torch.mean (torch.abs(pdeu))

        batch_gen=lambda size:default_batch_generator(size,[t_range,u_range,v_range,k_range])

        #trainer.add_loss(LOSS_PINN(FHN_LOSS,batch_gen,batch_size=64),weigth=1)



        ##BoundaryLoss

        def f(data_in, model):
                x,w = model(data_in).T 
                t,u,v,k=data_in.T          
                return torch.pow(torch.abs(x-u),2) 


        batch_gen=lambda size,de:default_batch_generator(size,[(0,0),u_range,v_range,k_range],de)
        #trainer.add_loss(LOSS_PINN(f,batch_gen,batch_size=1024,device=device),weigth=1)


        ##Validator
      #  trainer.add_validator(FHN_VAL_fromDataSet("training_data/validation/",device=device,name="val",dtype=dtype))
        trainer.add_validator(FHN_VAL_fromDataSet("training_data/treinocr2/",device=device,name="val",dtype=dtype))




        subprocess.Popen(f"python Results_plotter.py {base_dir}", shell=True, stdout=subprocess.PIPE).stdout.read()
        subprocess.Popen(f"python speed_com.py {base_Dir}", shell=True, stdout=subprocess.PIPE).stdout.read()






        trainer.train(1000)

        print(model)

