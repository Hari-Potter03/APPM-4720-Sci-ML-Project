# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torch.nn.functional as F
from denoising_diffusion_pytorch import Unet
from torch.optim.swa_utils import AveragedModel,get_ema_multi_avg_fn
#from diffusers import DDIMScheduler,DDPMScheduler
import os

# %%
if torch.cuda.is_available():
    device = torch.device("cuda")  # Use CUDA
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS
else:
    device = torch.device("cpu")  # Fallback to CPU

# %%
class catDataset(Dataset):
    def __init__(self, img_dir, second_img_dir):
        self.img_dir = img_dir
        self.img_names = []

        for filename in os.listdir(img_dir):
            file_path = os.path.join(img_dir, filename)
            if os.path.isfile(file_path):
                self.img_names.append(file_path)
        
        for filename in os.listdir(second_img_dir):
            file_path = os.path.join(second_img_dir, filename)
            if os.path.isfile(file_path):
                self.img_names.append(file_path)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = read_image(self.img_names[idx])
        image = image.to(torch.float32)/255.0
        return image

# %%
def get_model_size(model):
    return sum(p.numel() for p in model.parameters())


# %%
ds=catDataset("./cats","./mycat_64x64")
batch_size = 64
train_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# %%
class DDPMScheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

# %%
total_timesteps = 1000
sch = DDPMScheduler(total_timesteps)
#sch.set_timesteps(50)
# Inference and training are totally seperate
# train always is t -> t-1 
#sch.step(img1,25,)
# Use gaussian noise 
# img, gaussian, timestep
# sch.add_noise(img1,torch.randn_like(img1),torch.tensor(999))
# plt.imshow can work with floats just complains a bit
# .permute(1, 2, 0)

# %%
def train_loop(dataloader, model, loss_fn, optimizer, ema,e):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X) in enumerate(dataloader):
        x_0 = X.to(device)
        curr_bs = X.shape[0]
        t = torch.randint(0,total_timesteps,(curr_bs,))
        noise = torch.randn_like(x_0,requires_grad=False)

        a = sch.alpha[t].view(curr_bs,1,1,1).to(device)
        
        xin = (torch.sqrt(a)*x_0) + (torch.sqrt(1-a)*noise) # Noised image
        t=t.to(device)

        pred = model(xin,t) # Predict noise at t
        loss = loss_fn(noise,pred) # Difference between predicted noise and true

        # Backpropagation
        loss.backward()
        optimizer.step()
        if(e > 15):
            # Only update after first epoch to avoid all really bad params 
            ema.update_parameters(model)
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# %%
def do_infer(i,model):
    with torch.no_grad():
        x_t = torch.randn_like(ds[0]).unsqueeze(0).to(device)  # Initialize z on the correct device
        
        # Loop through timesteps in reverse order
        for t in reversed(range(0, total_timesteps)):
            t_tensor = torch.tensor(t).to(device).unsqueeze(0)
            temp = (sch.beta[t]/( (torch.sqrt(1-sch.alpha[t]))*(torch.sqrt(1-sch.beta[t])) ))
            x_t = (1/(torch.sqrt(1-sch.beta[t])))*x_t - (temp*model(x_t,t_tensor))
            e = torch.randn_like(x_t)
            x_t = x_t + (e*torch.sqrt(sch.beta[t]))
    
    plt.imshow(x_t.squeeze(0).permute(1,2,0).cpu().detach().numpy())
    plt.show()
    plt.savefig(f"{i}.png")

# %%
training = True
decay = 0.9999
if(training):
    model = Unet(
        dim=64,
        channels=3,
        dim_mults=[1,2,4,8,16],
        flash_attn=True,
    ).to(device)
    losf = nn.MSELoss()
    print(f"Model has {get_model_size(model)} parameters")
    epochs = 50
    opt = torch.optim.AdamW(model.parameters(),2e-5)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)
    for i in range(epochs):
        print(f"Epoch {i}")
        train_loop(train_dataloader,model,losf,opt,ema,i)
        if(i>15):
            do_infer(i,ema)
        else:
            do_infer(i,model)
    torch.save(ema.state_dict(),"ema_epc.pth")
    torch.save(model,"non_ema_epc.pth")
else:
    model = Unet(
        dim=64,
        channels=3,
        dim_mults=[1,2,4,8,16],
        flash_attn=True,
    ).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay)).to(device)
    ema.load_state_dict(torch.load("./ema_epc.pth",map_location=torch.device(device)))


# %%
images = []
with torch.no_grad():
    x_t = torch.randn_like(ds[0]).unsqueeze(0).to(device)  # Initialize z on the correct device
    times = [1,100,200,300,400,500,600,700,800,999]
    
    # Loop through timesteps in reverse order
    for t in reversed(range(0, total_timesteps)):
        t_tensor = torch.tensor(t).to(device).unsqueeze(0)
        temp = (sch.beta[t]/( (torch.sqrt(1-sch.alpha[t]))*(torch.sqrt(1-sch.beta[t])) ))
        x_t = (1/(torch.sqrt(1-sch.beta[t])))*x_t - (temp*model(x_t,t_tensor))
        e = torch.randn_like(x_t)
        x_t = x_t + (e*torch.sqrt(sch.beta[t]))

        if(t in times):
            images.append(x_t)

# %%
do_infer("bleh",model)

# %%
for i,image in enumerate(images):
    plt.imshow(image.squeeze(0).permute(1,2,0).cpu().detach().numpy())
    plt.show()
    plt.savefig(f"./img_at_step_{i*100}")


