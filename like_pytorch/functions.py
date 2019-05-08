import numpy as np

class Activation(object):
	""" Interface for activation functions """
	
	def __init__(self):
		# state is used to store values
		# stored values are useful in calculating derivatives
		self.state = None
		
	def __call__(self, x):
		return self.forward(x)
		
	def forward(self, x):
		raise NotImplemented
		
	def backward(self):
		raise NotImplemented
		
class Identity(Activation):
	"""Identity Activation"""
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		self.state = x.shape
		return x 
	
	def backward(self):
		return np.ones(self.state)

class ReLU(Activation):
	"""ReLU Activation"""
	def __init__(self):
		super(ReLU, self).__init__()
		
	def forward(self, x):
		out = np.maximum(0, x)
		self.state = (out > 0).astype(float)
		return out
		
	def backward(self):
		return self.state
		
class Sigmoid(Activation):
	"""Sigmoid Activation"""
	def __init__(self):
		super(Sigmoid, self).__init__()
		
	def forward(self, x):
		out = 1/(1+np.exp(-x))
		self.state = (1 - out) * out
		return out
		
	def backward(self):
		return self.state
		
###############################################################################

class Loss(object):
	"""Loss Function Interface"""
	
	def __init__(self):
		# state is used to store values
		# stored values are useful in calculating derivatives
		self.state = None
		
	def __call__(self, x, y):
		return self.forward(x, y)
		
	def forward(self, x, y):
		raise NotImplemented
		
	def backward(self):
		raise NotImplemented
		
class SoftmaxEntropy(Loss):
	"""Softmax Cross Entropy, takes logits"""
	
	def __init__(self, num_class):
		super(SoftmaxEntropy, self).__init__()
		self.num_class = num_class
	
	def forward(self, x, y):
		"""Convert y into onehot"""
		y = onehot_encode(y, self.num_class)
		
		"""Normalize logits to ensure precision"""
		mu = np.mean(x, 1).reshape(-1, 1)
		x = x - mu
		
		"""Calculate Softmax"""
		exps = np.exp(x)
		sums = np.sum(exps, 1).reshape(-1, 1)
		self.state = exps / sums - y
		
		"""Calculate Entropy"""
		entropy = np.sum(y * (np.log(sums) - x))
		return entropy
		
	def backward(self):
		return self.state
		
class MSELoss(Loss):
	"""Mean Square Loss"""
	
	def __init__(self):
		super(MSELoss, self).__init__()
	
	def forward(self, x, y):
		self.state = x - y
		return np.sum((x - y) ** 2)
		
	def backward(self):
		return 2 * self.state
			
###############################################################################
	
def onehot_encode(labels, num_class):
	onehot = np.zeros((len(labels),num_class))
	for row,label in enumerate(labels):
		onehot[row][label] = 1.0
	return onehot
	
def init_fn(input_size, output_size):
	bound = 2.0/np.sqrt(input_size)
	W = np.random.randn(input_size, output_size) * bound
	b = np.zeros(output_size)
	return (W,b)
	
def get_pos(img_shape, bsize, stride):

	assert (len(img_shape) == 4)
	_,m,n,D = img_shape
	
	assert((m-bsize[0])%stride == 0)
	assert((n-bsize[1])%stride == 0)
	orows = (m - bsize[0]) // stride + 1
	ocols = (n - bsize[0]) // stride + 1
	
	# get row map
	irow = np.repeat(np.arange(bsize[0]), bsize[1]*D)
	icol = stride * np.repeat(np.arange(orows), ocols)
	row_map = irow.reshape(1,-1) + icol.reshape(-1,1)
	
	# get col map
	jrow = np.tile(np.repeat(np.arange(bsize[1]), D), bsize[0])
	jcol = stride * np.tile(np.arange(ocols), orows)
	col_map = jrow.reshape(1,-1) + jcol.reshape(-1,1)

	# get depth map
	dep_map = np.arange(D)
	dep_map = np.tile(dep_map, bsize[0] * bsize[1] * orows * ocols)
	dep_map = dep_map.reshape(orows * ocols, -1)
	
	return row_map, col_map, dep_map
	
def im2col(A, maps):
	N = len(A)
	rows = len(maps[0])
	im2col = A[:, maps[0], maps[1], maps[2]].reshape(N*rows, -1)
	return im2col
	
def col2im(A, maps, shape):
	x = np.empty(shape)
	np.add.at(x, (slice(None), maps[0], maps[1], maps[2]), A)
	return x
	
def pad2d(A, padsize):
	s = padsize
	return np.pad(A, ((0,0), (s,s), (s,s), (0,0)), 'constant')
	