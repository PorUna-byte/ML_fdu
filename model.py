import torch
import torch.nn as nn
import deepspeed
import torch.nn.functional as F

# Define the model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Shared layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        # Classification head
        self.fc1 = nn.Linear(128 * 16 * 16, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

        # Segmentation head
        self.conv_trans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv_trans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.conv_trans3 = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.conv_trans4 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1)
        self.conv_trans5 = nn.ConvTranspose2d(16, 1, kernel_size=1)

    def forward(self, x):
        #(N,H,W,C)->(N,C,H,W)
        x = torch.permute(x,(0,3,1,2))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        # Classification head
        x_class = x.reshape(-1, 128 * 16 * 16)
        x_class = F.relu(self.fc1(x_class))
        x_class = self.dropout(x_class)
        x_class = self.sigmoid(self.fc2(x_class))

        # Segmentation head
        x_seg = F.relu(self.conv_trans1(x))
        x_seg = self.upsample(x_seg)
        x_seg = F.relu(self.conv_trans2(x_seg))
        x_seg = self.upsample(x_seg)
        x_seg = F.relu(self.conv_trans3(x_seg))
        x_seg = self.upsample(x_seg)
        x_seg = F.relu(self.conv_trans4(x_seg))
        x_seg = self.upsample(x_seg)
        x_seg = self.sigmoid(self.conv_trans5(x_seg))

        return x_class, torch.permute(x_seg, (0,2,3,1))


