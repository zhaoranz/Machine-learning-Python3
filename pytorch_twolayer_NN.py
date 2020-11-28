import torch
import torch.nn as nn
N, D_in, H, D_out=64,784, 100, 10

x=torch.randn(N, D_in)
y=torch.randn(N, D_out)
w1=torch.randn(D_in, H)
w2=torch.randn(H, D_out)
class TwoLayerNN(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    super(TwoLayerNN, self).__init__()
    self.linear1=torch.nn.Linear(D_in, H, bias=False)
    self.linear2=torch.nn.Linear(H, D_out, bias=False)
  def forward(self,x):
    y_pred=self.linear2(self.linear1(x).clamp(min=0))
    return y_pred
model=TwoLayerNN(D_in, H, D_out)
criterion=nn.MSELoss()
lr=1e-4
optimizer=torch.optim.Adam(model.parameters(),lr=lr)
for i in range(500):
  y_pred=model(x)
  loss=criterion(y_pred,y)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  if i %50==0:
    print(i,':',loss.item())
