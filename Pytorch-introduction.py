#A tutorial based on sentdex videos.
import torch
import torchvision
from torchvision import transforms, datasets
#download mnist dataset:
train=datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test=datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#may take a moment...


trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

#batch size is 1. for the large sample size, cannot fit any realistic example on GPU   2. generalization avoiding arbitrary optimazation

#visualized
import matplotlib.pyplot as plt

plt.imshow(data[0][0].view(28,29))
plt.show()

#make sure the dataset is balanced
total=0
counter_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}
for data in trainset:
  Xs, ys=data
  for y in ys:
    counter_dict[imt(y)]+=1
    total+=1
for i in counter_dict:
  print(f"{i}:{counter_dict[i]/total*100}")

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
  def __init__(self):
    super().__init__()#this is important
    self.fc1=nn.Linear(784, 64)
    self.fc2=nn.Linear(64, 64)
    self.fc3=nn.Linear(64, 64)
    self.fc4=nn.Linear(64, 10)
  def forward(self, x):
    x=F.relu(self.fc1(x))#why not set different activation functions for different neurons?
    x=F.relu(self.fc2(x))
    x=F.relu(self.fc2(x))
    x=self.fc4(x) 
    
    return F.log_softmax(x,dim=1)#dim-0, batches
    
net=Net()
print(net)

#some data:
x=torch.rand((28*28))
X=x.view(-1,28*28)
output=net(X)
    
    
    
    




