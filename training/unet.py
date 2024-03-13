import torch
import torch.nn as nn
import torch.nn.functional as F

from unet_block import lwh_1x1_conv,DoubleConv, Down, Up, OutConv

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear


        self.conv1x1 = (lwh_1x1_conv(n_channels, 32))
        self.inc = (DoubleConv(32, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x, mask=None):
        # large receptive feild(shrink image size) 
        # H -> H // 2 -> H // 4 -> H // 8 -> H // 16
        # minimum image size 16
        x = self.conv1x1(x)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if mask is not None:
            output = logits - (1-mask)*10**10
            return torch.sigmoid(output) # prob
        else:
            return torch.sigmoid(logits)

if __name__ == "__main__":
    
    in_channel = 3
    num_class = 1
    
    model = UNet(in_channel, num_class)
    model.eval()
    B, C, H, W = 1, in_channel, 50, 50
    
    test_data = torch.randn(B, C, H, W) 
    
    out = model(test_data)
    
    # mask = torch.zeros(B, 1, H, W, dtype=bool) # feasible mask
    # mask[0][0][0][0] = True   
    # print(mask)
    
    
    
    print(out.shape) # (B, num_class, H, W)