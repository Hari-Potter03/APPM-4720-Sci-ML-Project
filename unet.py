import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels+512, out_channels)

    def forward(self, x, skip, embedding,t):
        x = self.up(x)
        x = torch.cat([x, skip,embedding(x,t)], dim=1)
        x = self.conv_block(x)
        return x

class SineEmbedding(nn.Module):
    def __init__(self, time_steps:int, embedding_dim: int):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.time_steps = time_steps

        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_dim))
        embeddings = torch.zeros(time_steps, embedding_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(x.device)
        embeds = embeds.view(1, self.embedding_dim, 1, 1)
        embeds = embeds.expand(x.shape[0], self.embedding_dim, x.shape[2], x.shape[3])
        return embeds

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,max_timesteps=1000,embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_timesteps=max_timesteps

        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)

        self.bottleneck = ConvBlock(512, 1024)

        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        self.time_embedding = SineEmbedding(max_timesteps,embedding_dim)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.sig = nn.Sigmoid()
    def forward(self, x, t):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x,skip4,self.time_embedding,t)
        x = self.dec3(x,skip3,self.time_embedding,t)
        x = self.dec2(x,skip2,self.time_embedding,t)
        x = self.dec1(x,skip1,self.time_embedding,t)

        x = self.sig((self.final_conv(x)))
        return x

# # Example usage
# if __name__ == "__main__":
#     model = UNet(in_channels=3, out_channels=3)
#     input_tensor = torch.randn(16, 3, 64, 64)  # Batch size of 1, 3 channels, 128x128 image
#     output = model(input_tensor,torch.tensor(10))
#     print(output.shape)
