"""
CAP 5404 Deep Learning for Computer Graphics
Project II. Neural Networks & Computer Graphics

Pranath Reddy Kumbam (UFID: 8512-0977)

Code to show the output of our trained models
Program to Predict mean chrominance values and Colorize Grayscale images
Uses Models that achieved the best scores
"""

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import random
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Set Mode: Faces or NCD
mode = input('Enter Mode: [Face:1, NCD:2] ')

# Define Models
class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()

        self.regressor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.regressor(x)
        return x

class DCA(nn.Module):
    def __init__(self):
        super(DCA, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1600)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 2, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(-1,64,5,5)
        x = self.decoder(x)
        return x

# Import pre-trained models
if mode == '1':
    if torch.cuda.is_available():
        CNN_model = torch.load('./Models/CNN_Regressor_Faces_Tanh.pth')
        DCA_model = torch.load('./Models/DCA_Faces_Tanh.pth')
    else:
        CNN_model = torch.load('./Models/CNN_Regressor_Faces_Tanh.pth', map_location=torch.device('cpu'))
        DCA_model = torch.load('./Models/DCA_Faces_Tanh.pth', map_location=torch.device('cpu'))
else:
    if torch.cuda.is_available():
        CNN_model = torch.load('./Models/CNN_Regressor_NCD_Tanh.pth')
        DCA_model = torch.load('./Models/DCA_NCD_Tanh.pth')
    else:
        CNN_model = torch.load('./Models/CNN_Regressor_NCD_Tanh.pth', map_location=torch.device('cpu'))
        DCA_model = torch.load('./Models/DCA_NCD_Tanh.pth', map_location=torch.device('cpu'))

# Function to load images from test folder
def load_images_from_folder(folder):
  img_size = 128
  images = []
  for filename in os.listdir(folder):
    print('Importing ' + str(filename))
    if os.path.isdir(folder + filename):
      images.extend(load_images_from_folder(folder + filename))
    else:
      img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
      img = cv2.resize(img, (img_size, img_size))
      img = np.asarray(img).reshape(1,img.shape[0],img.shape[0]) / 255.0
      if img is not None:
        images.append(img)
  return images

# Load images
img_dir = "./Test" # Directory where the test images are stored
images = np.asarray(load_images_from_folder(img_dir))
images = torch.from_numpy(images.astype('float32'))
if torch.cuda.is_available():
    images = images.cuda()
reg_out = CNN_model(images)
reg_out = reg_out.detach().cpu().numpy()*255

# Regression
print('Mean chrominance Values Predicted by Regression Model')
print('Images ordered alphabetically')
for i in range(reg_out.shape[0]):
    print('a* : ' + str(reg_out[i][0]) + ', b* : ' + str(reg_out[i][1]))

# Colorize
color_out = DCA_model(images)
color_out = color_out.detach().cpu().numpy()
color_out = ((color_out/2)+0.5)*255
pred = np.concatenate((images.detach().cpu().numpy()*255, color_out), axis=1)
pred = np.transpose(pred, (0,2,3,1))
for i in range(reg_out.shape[0]):
    img_pred = cv2.cvtColor(pred[i].astype('uint8'), cv2.COLOR_LAB2RGB)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.imshow(images[i][0], cmap='gray')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.imshow(img_pred)
    ax1.title.set_text('Input grayscale image')
    ax2.title.set_text('Colorized image')
    plt.savefig(img_dir + '/Result_' + str(i+1) + '.png', format='png', dpi=300)
    plt.show()
