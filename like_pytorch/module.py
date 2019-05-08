import numpy as np
import functions as F

class Module(object):
	def __init__(self):
		raise NotImplemented
		
	def __call__(self, x):
		return self.forward(x)
		
	def forward(self, x):
		raise NotImplemented

	def backward(self, prop):
		raise NotImplemented
		
	def update(self, lr, momentum):
		raise NotImplemented
		
	def train(self):
		pass 
		
	def eval(self):
		pass

class Linear(Module):
	def __init__(self, input_size, output_size, decay=0.0, init_fn=None):
		if init_fn:
			self.W, self.B = init_fn(input_size, output_size)
		else:
			self.W, self.B = F.init_fn(input_size, output_size)
			
		self.dW = np.zeros_like(self.W)
		self.dB = np.zeros_like(self.B)
		
		# velocity vector is used for momentum
		self.vW = np.zeros_like(self.W)
		self.vB = np.zeros_like(self.B)
		
		self.decay = decay
		
	def forward(self, x):
		self.dW = x
		return np.matmul(x, self.W) + self.B
		
	def backward(self, prop):
		self.dW = np.matmul(self.dW.T, prop) / len(prop)
		self.dB = np.mean(prop, 0)
		prop = np.matmul(prop, self.W.T)
		return prop 
		
	def update(self, lr, momentum=0.0):
	
		self.vW = momentum * self.vW - lr * self.dW
		self.vB = momentum * self.vB - lr * self.dB
		
		self.W += self.vW
		self.B += self.vB
		
		self.W *= (1-self.decay)
		
class Conv2d(Module):
	def __init__(self, in_channel, out_channel, ker_size, stride, pad, 
	decay=0.0, init_fn=None):
		self.input_shape = None
		self.ker_size = ker_size 
		self.stride = stride 
		self.pad = pad 
		self.dout = out_channel
		self.X = None
	
		Wsize = in_channel * ker_size ** 2
		if init_fn:
			self.W, self.B = init_fn(Wsize, out_channel)
		else:
			self.W, self.B = F.init_fn(Wsize, out_channel)
			
		self.dW = np.zeros_like(self.W)
		self.dB = np.zeros_like(self.B)
		
		# velocity vector is used for momentum
		self.vW = np.zeros_like(self.W)
		self.vB = np.zeros_like(self.B)
		
		self.decay = decay
	
	def forward(self, x):
		ker_size = self.ker_size
		stride = self.stride
		pad = self.pad
		
		## allocate spaces for the output
		N,m,n,D = x.shape
		rows = (m + pad * 2 - ker_size)//stride+1
		cols = (n + pad * 2 - ker_size)//stride+1
		
		# pad the image
		if pad != 0:
			x = F.pad2d(x, pad)
			
		## store input shape for backpropgation
		self.input_shape = x.shape
		
		# get position map 
		maps = F.get_pos(x.shape, (ker_size,ker_size), stride)
		
		# store maps to do backpropgation
		self.maps = maps
		
		# get im2col
		self.X = F.im2col(x, maps)
		
		out = np.matmul(self.X, self.W) + self.B.reshape(1,-1)
		return out.reshape(N, rows, cols, -1)
		
	def backward(self, prop):
		N,_,_,Dnew = prop.shape
		
		## get db
		db = np.sum(prop, (0,1,2)) / N
		
		## get dw: row*col*D*Dnew
		prop = prop.reshape(-1, Dnew)
		self.X = np.transpose(self.X, (1,0))
		
		dW = np.matmul(self.X, prop) / N
		
		## get derivative of col
		dX_col = np.matmul(prop, self.W.T)
		## reshape to N,rows,window
		_,window = dX_col.shape
		dX_col = dX_col.reshape(N,-1,window)
		dX_pad = F.col2im(dX_col, self.maps, self.input_shape)
		
		pad = self.pad
		ker_size = self.ker_size
		return dX_pad[:,pad:-pad,pad:-pad,:] / (ker_size ** 2)
		
		
	def update(self, lr, momentum=0.0):
	
		self.vW = momentum * self.vW - lr * self.dW
		self.vB = momentum * self.vB - lr * self.dB
		
		self.W += self.vW
		self.B += self.vB
		
		self.W *= (1-self.decay)
		
class Maxpool2d(Module):
	def __init__(self, kersize):
		self.kersize = kersize
		self.weights = None
		
	def forward(self, x):
		s = self.kersize 
		N,m,n,D = x.shape
		assert(m%s == 0)
		assert(n%s == 0)
		
		## calculate the output rows
		orows = m // s
		ocols = n // s
		
		maps = F.get_pos(x.shape, (s,s), s)
		im2col = F.im2col(x, maps)
		rows,_ = im2col.shape
		im2col = im2col.reshape(rows,-1,D)
		im_max = np.max(im2col, 1)
		
		## store which elements are biggest
		self.weights = np.argmax(im2col, 1)
		
		return im_max.reshape(N,orows,ocols,D)
		
	def backward(self, prop):
		s = self.kersize
		expand_prop = np.repeat(np.repeat(prop, s, 1), s, 2)

		N,m,n,D = prop.shape 
		self.weights = np.transpose(self.weights, (1,0)).reshape(-1)
		expand_weights = np.zeros((D*N*m*n,s*s))
		expand_weights[np.arange(D*N*m*n), self.weights] = 1.0
		expand_weights = expand_weights.reshape(D,N,m,n,s,s)
		expand_weights = np.transpose(expand_weights,(1,2,4,3,5,0))
		expand_weights = expand_weights.reshape(N,m*s,n*s,D)
		
		return expand_weights * expand_prop
		
	def update(self, lr, momentum=0.0):
		pass
		
class Flatten(Module):
	def __init__(self):
		self.input_shape = None
		
	def forward(self, x):
		self.input_shape = x.shape 
		N = len(x)
		return x.reshape(N, -1)

	def backward(self, prop):
		return prop.reshape(self.input_shape)
		
	def update(self, lr, momentum):
		pass
		
class BatchNorm(Module):
	def __init__(self, size, alpha=0.9):
		self.train = True 
		
		# alpha determines how much running mean is used
		self.alpha = alpha
		# this is used to prevent divide by 0
		self.eps = 1e-8 
		
		# keep track of (running) mean and variance
		self.mean = np.zeros((1, size))
		self.running_mean = np.zeros((1, size))
		self.var = np.ones((1, size))
		self.running_var = np.ones((1, size))
		
		# two parameters, gamma * norm + beta
		# these two parameters will be trained
		self.gamma = np.ones((1, size))
		self.dgamma = np.zeros((1, size))

		self.beta = np.zeros((1, size))
		self.dbeta = np.zeros((1, size))
		
	def forward(self, x):
		# keep x stored to do back propgation
		self.x = x
		
		if self.train:
			# keep mean and variance stored to do back propgation
			self.mean = np.mean(x, 0).reshape((1,-1))
			self.var = np.var(x, 0).reshape((1,-1))
			
			self.running_mean += (1 - self.alpha) * (self.mean - self.running_mean)
			self.running_var += (1 - self.alpha) * (self.var - self.running_var)
			self.norm = (x - self.mean)/np.sqrt(self.var + self.eps)
		else:
			self.norm = (x - self.running_mean)/np.sqrt(self.running_var + self.eps)
			
		return np.multiply(self.gamma, self.norm) + self.beta
		
	def backward(self, prop):
		N = len(prop) 
		std = np.sqrt(self.var + self.eps)
		x_mu = self.x - self.mean
		
		self.dbeta = np.mean(prop, 0).reshape((1,-1))
		self.dgamma = np.mean(prop * self.norm, 0).reshape((1,-1))
		dnorm = prop * self.gamma
		
		dx_mu_p1 = 1.0/std
		dx_mu_p2 = - x_mu * np.mean(x_mu, 0).reshape((1,-1)) / (std ** 3)
		dx_mu = dnorm * (dx_mu_p1 + dx_mu_p2)
		
		dx = dx_mu - np.mean(dx_mu, 0).reshape((1,-1))
		return dx
		
	def update(self, lr, momentum=0.0):
		self.gamma = self.gamma - lr * self.dgamma
		self.beta = self.beta - lr * self.dbeta
		
	def train(self):
		self.train = True 
		
	def eval(self):
		self.train = False
		
class Dropout(Module):
	def __init__(self, p):
		self.p = p
		self.p_mat = None
		self.train = True
		
	def forward(self, x):
		if self.train:
			self.p_mat = (1.0/self.p) * np.random.binomial(1, self.p, x.shape)
			return self.p_mat * x
		else:
			return x
		
	def backward(self, prop):
		return prop * self.p_mat
		
	def update(self, lr, momentum=0.0):
		pass
		
	def train(self):
		self.train = True 
		
	def eval(self):
		self.train = False
		
###############################################################################

class ListModule(Module):
	def __init__(self, modules):
		self.modules = modules 
		
	def forward(self, x):
		for module in self.modules:
			x = module(x)
		return x
		
	def backward(self, prop):
		for module in reversed(self.modules):
			if isinstance(module, Module):
				prop = module.backward(prop)
			elif isinstance(module, F.Activation):
				prop = prop * module.backward()

	def update(self, lr, momentum=0.0):
		for module in self.modules:
			if isinstance(module, Module):
				module.update(lr, momentum)

	def __getitem__(self, idx):
		return self.modules[idx]
	