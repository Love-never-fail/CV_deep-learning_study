import numpy as np
import math

def sum_squares_error(y, t):
  return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
  delta = 1e-7
  return -np.sum(t * np.log(y+delta))

class AddLayer:
  def __init__(self):
    pass
  
  def forward(self, x, y):
    out = x + y
    return out
  
  def backward(self, dout):
    dx = dout * 1
    dy = dout * 1
    return dx, dy


class MulLayer:
  def __init__(self):
    self.x = None
    self.y = None
  
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x * y

    return out
  
  def backward(self, dout):
    dx = dout * self.y
    dy = dout * self.x

    return dx, dy
  
class Relu:
  def __init__(self):
    self.mask = None
  
  def forward(self, x):
    self.mask = (x<=0)
    out = x.copy()
    out[self.mask] = 0

    return out
  
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout

    return dx 

class Sigmoid:
  def __init__(self):
    self.out = None
  
  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out

    return out
  
  def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out

    return dx  

class Affine:
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None
  
  def forward(self, x):
    self.x = x
    out = np.dot(x, self.W) + self.b

    return out

  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)

    return dx, self.dW, self.db

def softmax(x):
  exp_a = np.exp(a)
  sum_exp_a = np.sum(exp_a)
  y = exp_a / sum_exp_a

  return y

def crossEntropyError(y, t):
  delta = 1e-7
  return -np.sum(t*np.log(y+delta))  # log는 밑이 e인 자연로그로 나옴

class SoftmaxwithLoss:
  def __init__(self):
    self.loss = None
    self.y = None
    self.t = None
  
  def forward(self, x, t):
    self.t = t
    self.y = softmax(x)
    self.loss = crossEntropyError(self.y, self.t)
    
    return self.y

  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size

    return dx
