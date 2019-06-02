"""
__author__  = Willy Fitra Hendria

Simple autoencoder with 2 hidden layers on each encoder and decoder.
The first hidden-layer will have 256 units, and the second 128 units.
Sigmoid used as the activation function.
The initial weight matrices and biases are randomly sampled from normal distribution.
Training by minimizing the average mean squared error.
Reconstructed image will be plotted after finishing the iterations.
"""

import os
import numpy
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image


NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

if not os.path.exists('./img'):
    os.mkdir('./img')
	
img_transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x

class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()
		self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
			nn.Sigmoid())
		self.decoder = nn.Sequential(
			nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.Linear(256, 28*28),
            nn.Sigmoid())
	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x
		
def weights_init_normal(m: nn.Module) -> None:
	classname = m.__class__.__name__
	if classname.startswith("Linear"):    
		m.weight.data.normal_(mean=0, std=1/numpy.sqrt(m.in_features))
		m.bias.data.normal_(mean=0, std=1/numpy.sqrt(m.in_features))
	return

model = Autoencoder().cuda()
model.apply(weights_init_normal)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

for epoch in range(NUM_EPOCHS):
    for data in dataloader:
        img, _ = data
        img = img.view(img.size(0), -1)
        img = Variable(img).cuda()
        # forward
        output = model(img)
        loss = criterion(output, img)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # print loss
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, NUM_EPOCHS, loss.data))

pic = to_img(output.cpu().data)
save_image(pic, './img/reconstructed_image.png'.format(epoch))

torch.save(model.state_dict(), './sim_autoencoder.pth')