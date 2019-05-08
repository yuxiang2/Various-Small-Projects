import numpy as np

def getBatch(data, labels, batch_size=32, random=True):
	assert len(data) == len(labels)
	
	size = len(data)
	if random:
		order = np.random.permutation(size)
	else:
		order = np.arange(size)
		
	index = 0
	while index < size:
		next_index = index + batch_size
		yield data[order[index: next_index]], labels[order[index: next_index]]
		index = next_index


def trainEpoch(model, loss_f, data, labels, lr=0.001, 
batch_size=32, train=True, momentum=0.0):

	if train:
		model.train()
	else:
		model.eval()
		
	loss = 0
	correct = 0
	total = 0
	
	for batch_x, batch_y in getBatch(data, labels, batch_size=batch_size):
		pred = model(batch_x)
		loss += loss_f(pred, batch_y)
		
		if train:
			prop = loss_f.backward()
			model.backward(prop)
			model.update(lr, momentum=momentum)
		
		# get accuracy
		total += len(pred)
		correct += countCorrect(pred, batch_y)
		
	loss /= len(data)
	
	return loss, float(correct)/total

def countCorrect(logits, labels):
	pred_labels = np.argmax(logits, 1)
	return np.sum(pred_labels == labels)