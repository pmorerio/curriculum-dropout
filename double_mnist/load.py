import skimage
import numpy as np
import random as rnd
import os
import matplotlib.pyplot as plt
import skimage

def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def mnist(datasets_dir='/data/datasets/',ntrain=60000,ntest=10000,onehot=True):
	data_dir = os.path.join(datasets_dir,'mnist/')
	fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trX = loaded[16:].reshape((60000,28,28)).astype(float)

	fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	trY = loaded[8:].reshape((60000))

	fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teX = loaded[16:].reshape((10000,28,28)).astype(float)

	fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
	loaded = np.fromfile(file=fd,dtype=np.uint8)
	teY = loaded[8:].reshape((10000))

	trX = trX/255.
	teX = teX/255.
	 
	trX = trX[:ntrain]
	trY = trY[:ntrain]

	teX = teX[:ntest]
	teY = teY[:ntest]

	if onehot:
		trY = one_hot(trY, 10)
		teY = one_hot(teY, 10)
	#else:
		#trY = np.asarray(trY)
		#teY = np.asarray(teY)

	return trX,teX,trY,teY
	
def doubleMnist(datasets_dir='/data/datasets/'):
	#rndSeed=231
	#rnd.seed(rndSeed)
	print 'Generating '
	trX,teX,trY,teY = mnist(datasets_dir,onehot=False)

	train_size = 60000
	test_size = 10000
	
	trainX = np.zeros((train_size,64,64))
	trainY = np.zeros((train_size,55))
	testX = np.zeros((test_size,64,64))
	testY = np.zeros((test_size,55))

	for ii  in range(len(trainY)):
		
		xc1,yc1 = rnd.randint(0,35), rnd.randint(0,35)
		xc2,yc2 = rnd.randint(0,35), rnd.randint(0,35)
		
		i1 = rnd.randint(0,60000-1)
		i2 = rnd.randint(0,60000-1)
		trainX[ii,yc1:yc1+28,xc1:xc1+28] += trX[i1,:,:]
		trainX[ii,yc2:yc2+28,xc2:xc2+28] += trX[i2,:,:]
		a = np.max((trY[i1],trY[i2]))
		b = np.min((trY[i1],trY[i2]))
		lab = (a+1)*(a+2)/2 -a + b - 1 
		trainY[ii,:] = one_hot(lab,55)
	
	print '... '
		
	for ii  in range(len(testY)):
		
		xc1,yc1 = rnd.randint(0,35), rnd.randint(0,35)
		xc2,yc2 = rnd.randint(0,35), rnd.randint(0,35)
		
		i1 = rnd.randint(0,10000-1)
		i2 = rnd.randint(0,10000-1)
		testX[ii,yc1:yc1+28,xc1:xc1+28] += teX[i1,:,:]
		testX[ii,yc2:yc2+28,xc2:xc2+28] += teX[i2,:,:]
		a = np.max((teY[i1],teY[i2]))
		b = np.min((teY[i1],teY[i2]))
		lab = (a+1)*(a+2)/2 -a + b -1 
		testY[ii,:] = one_hot(lab,55)
	
	print 'Done!'
	return trainX, trainY, testX, testY
		
def main():

	data_ = open('../data_dir.txt','r')
	datasets_dir = data_.readline().split()[0]
	double_mnist, labels, _, _= doubleMnist(datasets_dir)	
	print 'label', labels[110,:]
	plt.imshow(double_mnist[110,:,:])
	plt.show()

	

if __name__ == "__main__":
	main()
