# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from denoising_diffusion_pytorch import Unet
#from diffusers import DDIMScheduler,DDPMScheduler
import os
from unet import UNet

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

    def forward(self, t):
        return self.beta[t], self.alpha[t]

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
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        # Initialize EMA parameters with the same values as model parameters
        self.ema_params = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.ema_params[name] = self.decay * self.ema_params[name] + (1 - self.decay) * param

    def apply(self):
        for name, param in self.model.named_parameters():
            param.data = self.ema_params[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.ema_params[name]


# %%
def train_loop(dataloader, model, loss_fn, optimizer, ema):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X) in enumerate(dataloader):
        x_0 = X.to(device)
        curr_bs = X.shape[0]
        t = torch.randint(0,total_timesteps,(curr_bs,))
        noise = torch.randn_like(x_0,requires_grad=False)
        a = sch.alpha[t].view(curr_bs,1,1,1).to(device)
        x = (torch.sqrt(a)*x_0) + (torch.sqrt(1-a)*noise)
        t=t.to(device)

        pred = model(x,t)
        loss = loss_fn(noise,pred)

        # Backpropagation
        loss.backward()
        optimizer.step()
        ema.update()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# %%
training = False
if(training):
    model = Unet(
        dim=64,
        channels=3,
        dim_mults=[1,2,4,8],
        flash_attn=True,
    ).to(device)
    losf = nn.MSELoss()
    ema = EMA(model,decay=0.995)
    print(f"Model has {get_model_size(model)} parameters")
    epochs = 1
    opt = torch.optim.AdamW(model.parameters(),2e-5)
    for i in range(epochs):
        print(f"Epoch {i}")
        train_loop(train_dataloader,model,losf,opt,ema)
else:
    model = torch.load("./1epc.pth")

# %%
ema.apply()
torch.save(model,"1epc.pth")

# %%
images = []
with torch.no_grad():
    ema.apply()
    # Initialize z
    z = torch.randn_like(ds[0]).to(device).unsqueeze(0)
    
    # Loop through timesteps in reverse order
    for t in reversed(range(1, total_timesteps)):
        t = [t]
        temp = (sch.beta[t]/( (torch.sqrt(1-sch.alpha[t]))*(torch.sqrt(1-sch.beta[t]))))

        z = (1/(torch.sqrt(1-sch.beta[t])))*z - (temp*model(z,torch.tensor(t).to(device)))

        e = torch.randn_like(ds[0]).to(device).unsqueeze(0)
        z = z + (e*torch.sqrt(sch.beta[t]))
        temp = sch.beta[0]/( (torch.sqrt(1-sch.alpha[0]))*(torch.sqrt(1-sch.beta[0])) )
        x = (1/(torch.sqrt(1-sch.beta[0])))*z - (temp*model(z,[0]))

# %%
images = []
with torch.no_grad():
    ema.apply()  # Apply EMA parameters to the model if applicable
    z = torch.randn_like(ds[0]).unsqueeze(0).to(device)  # Initialize z on the correct device
    
    # Loop through timesteps in reverse order
    for t in reversed(range(0, total_timesteps)):
        t_tensor = torch.tensor([t], device=device)  # Ensure timestep is a tensor on the correct device
        
        # Compute temp
        temp = (sch.beta[t] / (torch.sqrt(1 - sch.alpha[t]) * torch.sqrt(1 - sch.beta[t])))

        # Update z
        z = (z / torch.sqrt(1 - sch.beta[t])) - (temp * model(z, t_tensor))

        if(t%100 == 0):
            images.append(z)

# %%
for i,image in enumerate(images):
    plt.imshow(image.squeeze(0).permute(1,2,0).cpu().detach().numpy())
    plt.show()
    plt.savefig(f"./img_at_step_{i*100}")

