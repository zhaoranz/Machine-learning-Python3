#A tutorial based on sentdex videos.
import torch
import torchvision
from torchvision import transforms, datasets

train=datasets.MNIST('', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

test=datasets.MNIST('', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#may take a moment...


trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

#batch size is for the large-scle data



