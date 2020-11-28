import torch
N,D_in, H, D_out=64,784, 100, 10
x=torch.randn(N, D_in)
y=torch.randn(N, D_out)
w1=torch.randn(D_in, H)
w2=torch.randn(H, D_out)
for iter in range(1000):
  h=x.mm(w1)
  h_relu=h.clamp(min=0)
  y_pred=h_relu.mm(w2)

  loss=(y_pred-y).pow(2).sum().item() #we run NN manually

  g_y_pred=2.*(y_pred-y)
  g_w2=h_relu.t().mm(g_y_pred)
  g_hrelu=g_y_pred.mm(w2.t())
  g_h=g_hrelu.clone()
  g_h[h<0]=0
  g_w1=x.t().mm(g_h)

  w1-=lr*g_w1
  w2-=lr*g_w2

  if iter %50==0:
    print(iter,':',loss)
