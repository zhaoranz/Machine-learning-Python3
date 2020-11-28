import numpy as np
N,D_in, H, D_out=64,784, 100, 10
X=np.random.randn(N,D_in)
Y=np.random.randn(N, D_out)
w1=np.random.randn(D_in, H)
w2=np.random.randn(H, D_out)
lr=1e-6
for iter in range(1000):
  h=X.dot(w1)
  h_relu=np.maximum(h,0)
  y_pred= h_relu.dot(w2)
  loss=np.square(y_pred-Y).sum()

  g_y_pred=2.*(y_pred-Y)
  g_w2=h_relu.T.dot(g_y_pred)
  g_hrelu=g_y_pred.dot(w2.T)
  g_h=g_hrelu.copy()
  g_h[h<0]=0
  g_w1=X.T.dot(g_h)

  w1-=lr*g_w1
  w2-=lr*g_w2

  if iter%100==0:
    print(iter,':',loss)


