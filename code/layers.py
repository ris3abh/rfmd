from abc import ABC, abstractmethod 
import numpy as np
import math

class Layer (ABC) :
    def init (self): 
        self . prevIn = [] 
        self. prevOut=[]

    def setPrevIn(self ,dataIn): 
        self . prevIn = dataIn

    def setPrevOut( self , out ): 
        self . prevOut = out

    def getPrevIn( self ):
        return self . prevIn

    def getPrevOut( self ):
        return self . prevOut

    @abstractmethod
    def forward(self ,dataIn):
        pass

    @abstractmethod
    def gradient(self):
        pass

    @abstractmethod
    def backward( self , gradIn ):
        pass

class inputLayer(Layer):
    def __init__(self, dataIn, zscore = True):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0, ddof = 1)
        self.stdX[self.stdX == 0] = 1
        self.zscore = True

    def forward(self, dataIn, zscore = True):
        if zscore:
            self.setPrevIn(dataIn)
            dataOut = (dataIn - self.meanX)/self.stdX
            self.setPrevOut(dataOut)
        else:
            self.setPrevIn(dataIn)
            dataOut = dataIn
            self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        pass

    def backward(self, gradIn):
        return gradIn*self.gradient()

## diagonal gradient, so hadamarts product
class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut =  dataIn
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        return np.identity(self.getPrevOut().shape[1])
        
    def backward(self, gradIn):
        pass

## diagonal gradient, so hadamarts product
class ReLULayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.maximum(0, dataIn)
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        prevOut = self.getPrevOut()
        return np.where(prevOut > 0, 1, 0)

    def backward(self, gradIn):
        grad = self.gradient()
        return np.multiply(gradIn, grad)

## non diagonal gradient, so tensor product
class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = self.softmax(dataIn)
        self.setPrevOut(dataOut)
        return self.getPrevOut()
        
    def gradient(self):
        T = []
        for row in self.getPrevOut():
            grad = np.diag(row) - row[np.newaxis].T.dot(row[np.newaxis])
            T.append(grad)
        return np.array(T)

    def backward(self, gradIn):
        grand = self.gradient()
        return np.einsum('ijk,ik->ij', grand, gradIn)

## diagonal gradient, so hadamarts products
class LogisticSigmoid(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = 1/(1+np.exp(-dataIn))
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        prevOut = self.getPrevOut()
        return self.getPrevOut() * (1 - self.getPrevOut())

    def backward(self, gradIn):
        grad = self.gradient()
        return np.multiply(gradIn, grad)

## diagonal gradient, so hadamarts product
class TanhLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.tanh(dataIn)
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        prevOut = self.getPrevOut()
        return 1 - np.square(prevOut)

    def backward(self, gradIn):
        grad = self.gradient()
        return np.multiply(gradIn, grad)
    
class dropOutLayer(Layer):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob
        self.mask = None

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.mask = np.random.binomial(1, self.keep_prob, size = dataIn.shape)
        dataOut = np.multiply(dataIn, self.mask)
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        return self.mask

    def backward(self, gradIn):
        grad = self.gradient()
        return np.multiply(gradIn, grad)

class SquaredErrorLoss():
    def __init__(self):
        super().__init__()
        self.y = None
        self.yhat = None

    def eval(self, y, yhat):
        self.y = y
        self.yhat = yhat
        return np.mean((y - yhat)*(y - yhat))

    def gradient(self, y, yhat):
        return 2*(yhat - y)

class NegativeLikelihood():
    def __init__(self):
        super().__init__()
        self.y = None
        self.yhat = None

    def eval(self, y , yhat):
        epsilon = 0.0000001
        self.y = y
        self.yhat = yhat
        yhat = np.clip(yhat, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        ## should return a single value

    def gradient(self, y, yhat):
        epsilon = 0.0000001
        return - (y - yhat)/ (yhat * (1 - yhat) + epsilon)

class CrossEntropyLoss():
    def __init__(self):
        super().__init__()
        self.y = None
        self.yhat = None

    def eval(self, y, yhat):
        epsilon = 0.0000001
        self.y = y
        self.yhat = yhat
        return -np.mean(np.sum(np.multiply(y, np.log(yhat + epsilon)), axis=1))

    def gradient(self, y, yhat):
        epsilon = 0.0000001
        return -np.divide(y, yhat + epsilon)
    

class L2Regularisation(Layer):
    def __init__(self, lamda):
        super().__init__()
        self.lamda = lamda

    def eval(self, error, weights):
        return error + self.lamda * np.sum(np.square(weights))
    
    def gradient(self, error, weights):
        return error + 2 * self.lamda * weights
        
class fullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut) -> None:
        super().__init__()
        self.sizeIn = sizeIn
        self.sizeOut = sizeOut
        # self.weights = np.random.uniform(low=-0.0001, high=0.0001, size = (sizeIn, sizeOut))
        # self.baises = np.random.uniform(low=-0.0001, high=0.0001, size = (1, sizeOut))
        ## Xavier initialization
        self.weights = np.random.uniform(low=-np.sqrt(6/(sizeIn + sizeOut)), high=np.sqrt(6/(sizeIn + sizeOut)), size = (sizeIn, sizeOut))
        self.baises = np.random.uniform(low=-np.sqrt(6/(sizeIn + sizeOut)), high=np.sqrt(6/(sizeIn + sizeOut)), size = (1, sizeOut))

        ## for ADAM
        self.biasS, self.biasR = 0, 0
        self.weightS, self.weightR = 0, 0
      
    def getWeights(self):
        return self.weights

    def getBaises(self):
        return self.baises

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.dot(dataIn, self.weights) + self.baises
        self.setPrevOut(dataOut)
        return self.getPrevOut()

    def gradient(self):
        ## prevOut = self.getPrevOut()
        return self.getWeights().T

    def update_weights(self, gradIn, learning_rate, epoch, adam = False, batch_size = 1):
        self.dj_db = np.sum(gradIn, axis = 0)/gradIn.shape[0]
        self.dj_dw = np.dot(self.getPrevIn().T, gradIn)/gradIn.shape[0]
        if adam:
            p1, p2 = 0.9, 0.999
            eta = math.pow(10, -8)
            self.weightS = (p1 * self.weightS) + ((1 - p1) * self.dj_dw)
            self.weightR = (p2 * self.weightR) + ((1 - p2) * (self.dj_dw * self.dj_dw))
            self.biasS = (p1 * self.biasS) + ((1 - p1) * self.dj_db)
            self.biasR = (p2 * self.biasR) + ((1 - p2) * (self.dj_db * self.dj_db))
            self.weights-= learning_rate * (self.weightS/(1 - math.pow(p1, epoch)))/(np.sqrt(self.weightR/(1 - math.pow(p2, epoch))) + eta)
            self.baises-= learning_rate * (self.biasS/(1 - math.pow(p1, epoch)))/(np.sqrt(self.biasR/(1 - math.pow(p2, epoch))) + eta)
        else:
            self.weights-= learning_rate * self.dj_dw
            self.baises-= learning_rate * self.dj_db

    def backward(self, gradIn):
        grad = self.gradient()
        return np.dot(gradIn, grad) 

class MaxPooling2D(Layer):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size
        self.indices = None

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.zeros((dataIn.shape[0], dataIn.shape[1]//self.size, dataIn.shape[2]//self.size))
        self.indices = np.zeros((dataIn.shape[0], dataIn.shape[1]//self.size, dataIn.shape[2]//self.size))
        for i in range(dataIn.shape[0]):
            for j in range(dataIn.shape[1]//self.size):
                for k in range(dataIn.shape[2]//self.size):
                    dataOut[i][j][k] = np.max(dataIn[i][j*self.size:(j+1)*self.size, k*self.size:(k+1)*self.size])
                    self.indices[i][j][k] = np.argmax(dataIn[i][j*self.size:(j+1)*self.size, k*self.size:(k+1)*self.size])
        self.setPrevOut(dataOut)
        return self.getPrevOut()
    
    def gradient(self):
        return self.getWeights().T
    
    def backward(self, gradIn):
        grad = np.zeros((gradIn.shape[0], gradIn.shape[1]*self.size, gradIn.shape[2]*self.size))
        for i in range(gradIn.shape[0]):
            for j in range(gradIn.shape[1]):
                for k in range(gradIn.shape[2]):
                    grad[i][j*self.size:(j+1)*self.size, k*self.size:(k+1)*self.size][int(self.indices[i][j][k])] = gradIn[i][j][k]
        return grad
    
class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.shape = None

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.shape = dataIn.shape
        dataOut = dataIn.reshape(dataIn.shape[0], -1)
        self.setPrevOut(dataOut)
        return self.getPrevOut()
    
    def gradient(self):
        return self.getWeights().T
    
    def backward(self, gradIn):
        return gradIn.reshape(self.shape)
    
class ConvolutionalLayer2D(Layer):
    def __init__(self, num_filters, filter_size, stride=1, padding=0, kernel = True):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.prevIn = []
        self.prevOut = []
        self.kernel = kernel
        self.filter = None

    ## taking kernel from the user 
    def setFilters(self, filters):
        self.filters = filters

    def getFilters(self):
        return self.filters

    def forward(self, dataIn):
    # Add an extra dimension for the number of data points
        dataIn = np.expand_dims(dataIn, axis=0)
        self.setPrevIn(dataIn)
        num_data, in_channels, in_height, in_width = dataIn.shape
        ## self.filters = np.random.randn(self.num_filters, in_channels, self.filter_size, self.filter_size)
        self.filter = self.getFilters()
        out_height = int((in_height + 2 * self.padding - self.filter_size) / self.stride + 1)
        out_width = int((in_width + 2 * self.padding - self.filter_size) / self.stride + 1)
        out = np.zeros((num_data, self.num_filters, out_height, out_width))
        padded_input = np.pad(dataIn, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')
        for data_idx in range(num_data):
            for filter_idx in range(self.num_filters):
                for h_idx in range(out_height):
                    for w_idx in range(out_width):
                        h_start = h_idx * self.stride
                        h_end = h_start + self.filter_size
                        w_start = w_idx * self.stride
                        w_end = w_start + self.filter_size
                        input_slice = padded_input[data_idx, :, h_start:h_end, w_start:w_end]
                        out[data_idx, filter_idx, h_idx, w_idx] = np.sum(input_slice * self.filters[filter_idx])        
        self.setPrevOut(out)
        return out

    def gradient(self):
        return

    def backward(self, gradIn):
        num_data, in_channels, in_height, in_width = self.getPrevIn().shape
        gradOut = np.zeros((num_data, in_channels, in_height, in_width))
        padded_input = np.pad(self.getPrevIn(), ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)), mode='constant')

        for data_idx in range(num_data):
            for filter_idx in range(self.num_filters):
                for h_idx in range(gradIn.shape[2]):
                    for w_idx in range(gradIn.shape[3]):
                        h_start = h_idx * self.stride
                        h_end = h_start + self.filter_size
                        w_start = w_idx * self.stride
                        w_end = w_start + self.filter_size
                        input_slice = padded_input[data_idx, :, h_start:h_end, w_start:w_end]
                        gradOut[data_idx, :, h_start:h_end, w_start:w_end] += gradIn[data_idx, filter_idx, h_idx, w_idx] * self.filters[filter_idx]

        self.setPrevOut(gradOut)
        return gradOut

X = np.array([[[1, 1, 0, 1, 0, 0, 1, 0],
               [1, 1, 1, 1, 0, 0, 1, 0],
               [0, 0, 1, 1, 0, 1, 0, 1],
               [1, 1, 1, 0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 0, 0, 1],
               [1, 0, 1, 0, 0, 1, 0, 1]]])

kernel = np.array([[2, -1, 2],
                   [2, -1, 0],
                   [1, 0, 2]])

L1 = ConvolutionalLayer2D(1, 3, 1, 0)
L1.setFilters(kernel)

print(L1.forward(X))