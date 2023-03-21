from abc import ABC, abstractmethod
import numpy as np

EPSILON = 1e-7

class Layer(ABC):
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self,dataIn):
        self.__prevIn = dataIn

    def setPrevOut(self, out):
        self.__prevOut = out

    def getPrevIn(self):
        return self.__prevIn

    def getPrevOut(self):
        return self.__prevOut
 
    def backward(self, gradIn):
        grad = self.gradient()
        return np.array([np.dot(gradIn_i, grad_i) for gradIn_i, grad_i in zip(gradIn, grad)])

    @abstractmethod
    def forward(self,dataIn):
        pass

    @abstractmethod  
    def gradient(self):
        pass

class InputLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()
        self.meanX = np.mean(dataIn, axis=0)
        self.stdX = np.std(dataIn, axis=0, ddof = 1)
        self.stdX[self.stdX == 0] = 1
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        zscored = (dataIn - self.meanX) / self.stdX
        self.setPrevOut(zscored)
        return zscored

    def gradient(self):
        pass

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(dataIn)
        return dataIn

    def gradient(self):
        return np.identity(self.getPrevIn().shape[1])

    def backward(self, gradIn):
        return gradIn

class ReluLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        dataOut = np.maximum(0, dataIn)
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        grad = np.where(self.getPrevOut() > 0, 1, 0)
        tensor = grad
        return tensor

    def backward(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(1 / (1 + np.exp(-dataIn)))
        return self.getPrevOut()
    
    def gradient(self):
        diag = self.getPrevOut() * (1 - self.getPrevOut()) + EPSILON
        return np.eye(len(self.getPrevOut()[0])) * diag[:, np.newaxis]

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        max = np.max(dataIn, axis = 1)[:, np.newaxis]
        self.setPrevOut(np.exp(dataIn - max) / np.sum(np.exp(dataIn - max), axis=1, keepdims=True))
        return self.getPrevOut()

    def gradient(self):
        out = self.getPrevOut()
        tensor = np.empty((0, out.shape[1], out.shape[1]))
        for row in out:
            grad = -(row[:, np.newaxis])*row
            np.fill_diagonal(grad, row*(1-row))
            tensor = np.append(tensor, grad[np.newaxis], axis = 0)
        return tensor
        
class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        dataInCal = np.where(dataIn > 100, 99, dataIn)
        dataInCal = np.where(dataInCal < -100, -99, dataInCal)
        dataOut = (np.exp(dataInCal) - np.exp(-dataInCal)) / (np.exp(dataInCal) + np.exp(-dataInCal))
        self.setPrevOut(dataOut)
        return dataOut

    def gradient(self):
        a = .000000000000001
        tensor = (1 - self.getPrevOut()**2) + a
        tensor = np.array(tensor)
        return tensor

    def backward(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut

class FullyConnectedLayer(Layer):
    def __init__(self, sizeIn, sizeOut, xavier_init = True, he_init = False):
        super().__init__()
        if xavier_init:
            self.xavier_init(sizeIn, sizeOut)
        elif he_init:
            self.he_init(sizeIn, sizeOut)
        else:
            self.weights = np.random.uniform(-0.001, 0.001, (sizeOut, sizeIn)).T
            self.biases = np.random.uniform(-0.001, 0.001, (1, sizeOut))

        # accumulators for Adam optimizer
        self.weights_s = 0
        self.weights_r = 0
        self.biases_s = 0
        self.biases_r = 0

        self.decay_1 = 0.9
        self.decay_2 = 0.999
        self.stability = 10e-8

    def xavier_init(self, sizeIn, sizeOut):
        bound = np.sqrt(6/(sizeIn+sizeOut))
        self.weights = np.random.uniform(-bound, bound, (sizeOut, sizeIn)).T
        self.biases = np.random.uniform(-bound, bound, (1, sizeOut))

    def he_init(self, sizeIn, sizeOut):
        mean = 0
        std_dev1 = np.sqrt(2/(sizeIn))
        std_dev2 = np.sqrt(2/1)
        self.weights = np.random.normal(mean, std_dev1, (sizeIn, sizeOut))
        self.biases = np.random.normal(mean, std_dev2, (1, sizeOut))

    def getWeights(self):
        return self.weights
    
    def setWeights(self, weights):
        self.weights = weights
    
    def getBiases(self):
        return self.biases
    
    def setBiases(self, biases):
        self.biases = biases

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(np.dot(dataIn, self.weights) + self.biases)
        return self.getPrevOut()

    def gradient(self):
        return np.array([self.weights.T for i in range(len(self.getPrevIn()))])
    
    def updateWeights(self, gradIn, epoch, learning_rate = 0.0001):
        dJdw = (self.getPrevIn().T @ gradIn) / gradIn.shape[0]
        self.weights_s = self.decay_1 * self.weights_s + (1 - self.decay_1) * dJdw
        self.weights_r = self.decay_2 * self.weights_r + (1 - self.decay_2) * dJdw * dJdw
        weights_update = (self.weights_s/(1-self.decay_1**(epoch+1))) / (np.sqrt(self.weights_r/(1-self.decay_2**(epoch+1))) + self.stability)
        
        dJdb = np.sum(gradIn, axis=0)/gradIn.shape[0]
        self.biases_s = self.decay_1 * self.biases_s + (1 - self.decay_1) * dJdb
        self.biases_r = self.decay_2 * self.biases_r + (1 - self.decay_2) * dJdb * dJdb
        biases_update = (self.biases_s/(1-self.decay_1**(epoch+1))) / (np.sqrt(self.biases_r/(1-self.decay_2**(epoch+1))) + self.stability)
        
        self.setWeights(self.getWeights() - learning_rate * weights_update)
        self.setBiases(self.getBiases() - learning_rate * biases_update)

class Conv2DLayer(Layer):
    def __init__(self, filters, kernel_size, stride=1, padding=0):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.kernel = self.init_kernel()
        self.stride = stride
        self.padding = padding

        # accumulators for Adam optimizer
        self.weights_s = 0
        self.weights_r = 0
        self.biases_s = 0
        self.biases_r = 0

        self.decay_1 = 0.9
        self.decay_2 = 0.999
        self.stability = 10e-8

    def init_kernel(self):
        bound = np.sqrt(6/(self.filters*self.kernel_size[0]*self.kernel_size[1]))
        return np.random.uniform(-bound, bound, (self.filters, self.kernel_size[0], self.kernel_size[1]))

    def getKernel(self):
        return self.kernel
    
    def setKernel(self, kernel):
        self.kernel = kernel
    
    def getPadding(self):
        return self.padding
        
    def setPadding(self, padding):
        self.padding = padding
    
    def getStride(self):
        return self.stride
    
    def setStride(self, stride):
        self.stride = stride

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(self.convolve(dataIn, self.kernel, self.padding, self.stride))
        return self.getPrevOut()

    def convolve(self, dataIn, kernel, padding, stride):
        return np.array([[self.convolve2D(dataIn_i, kernel_i, padding, stride) for kernel_i in kernel] for dataIn_i in dataIn])

    def convolve2D(self, dataIn, kernel, padding=0, stride=1):
        kernelHeight = kernel.shape[0]
        kernelWidth = kernel.shape[1]
        dataHeight = dataIn.shape[0]
        dataWidth = dataIn.shape[1]

        outputWidth = int(((dataWidth - kernelWidth + 2 * padding) / stride) + 1)
        outputHeight = int(((dataHeight - kernelHeight + 2 * padding) / stride) + 1)
        output = np.zeros((outputHeight, outputWidth))

        if padding != 0:
            dataInPadded = np.zeros((dataIn.shape[0] + padding*2, dataIn.shape[1] + padding*2))
            dataInPadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = dataIn
        else:
            dataInPadded = dataIn

        for y in range(outputHeight):
            for x in range(outputWidth):
                output[y, x] = (kernel * dataInPadded[y*self.stride: y*self.stride + kernelHeight, x*self.stride: x*self.stride + kernelWidth]).sum()

        # # return output
        # kh, kw = kernel.shape
        # dh, dw = dataIn.shape

        # # calculate output dimensions
        # oh = (dh - kh + 2 * padding) // stride + 1
        # ow = (dw - kw + 2 * padding) // stride + 1

        # # create padded data
        # if padding != 0:
        #     pad_dims = ((padding, padding), (padding, padding))
        #     dataIn = np.pad(dataIn, pad_dims)

        # # create a view of the input data with the appropriate shape and strides
        # sh, sw = dataIn.strides
        # shape = (oh, ow, kh, kw)
        # strides = (sh * stride, sw * stride, sh, sw)
        # data_view = np.lib.stride_tricks.as_strided(dataIn, shape=shape, strides=strides)

        # # apply element-wise multiplication and summation to get the convolution result
        # output = np.tensordot(data_view, kernel, axes=[(2, 3), (0, 1)])

        return output

    def gradient(self):
        return np.array([np.transpose(self.kernel, (0, 2, 1))]*len(self.getPrevIn()))

    def backward(self, gradIn):
        grad = self.gradient()
        return np.array([self.backward2D(grad_i, gradIn_i) for gradIn_i, grad_i in zip(gradIn, grad)])
        
    def backward2D(self, grad, gradIn):
        return np.array([self.convolve2D(np.pad(gradIn_i, self.kernel_size[0]-1, constant_values=0), grad_i) for gradIn_i, grad_i in zip(gradIn, grad)])

    def updateKernel(self, gradIn, epoch, learning_rate = 0.0001):
        # for gradIn_i in gradIn:
        #     dJdw = self.convolve2D(self.getPrevIn(), gradIn_i, padding=0, stride=1)
        #     self.weights_s = self.decay_1 * self.weights_s + (1 - self.decay_1) * dJdw
        #     self.weights_r = self.decay_2 * self.weights_r + (1 - self.decay_2) * dJdw * dJdw
        #     weights_update = (self.weights_s / (1 - self.decay_1**(epoch+1))) / (np.sqrt(self.weights_r / (1 - self.decay_2**(epoch+1))) + self.stability)
        #     self.setKernel(self.getKernel() - learning_rate * weights_update)
        for gradIn_i in gradIn:
            for dataIn_i in self.getPrevIn():
                for gradIn_i_kernel in gradIn_i:
                    self.updateKernel2D(gradIn_i_kernel, dataIn_i, epoch, learning_rate)

    def updateKernel2D(self, gradIn, dataIn, epoch, learning_rate = 0.0001):
        dJdw = np.array([self.convolve2D(dataIn, gradIn, padding=0, stride=1)])
        self.weights_s = self.decay_1 * self.weights_s + (1 - self.decay_1) * dJdw
        self.weights_r = self.decay_2 * self.weights_r + (1 - self.decay_2) * dJdw * dJdw
        weights_update = (self.weights_s/(1-self.decay_1**(epoch+1))) / (np.sqrt(self.weights_r/(1-self.decay_2**(epoch+1))) + self.stability)
        self.setKernel(self.getKernel() - learning_rate * weights_update)

class Conv3DLayer(Conv2DLayer):
    def __init__(self, filters, kernel_size, stride=1, padding=0):
        super().__init__(filters, kernel_size, stride, padding)
    
    def convolve(self, dataIn, kernel, padding, stride):
        return np.array([self.convolve3D(dataIn_i, kernel, padding, stride) for dataIn_i in dataIn])
    
    def convolve3D(self, dataIn, kernel, padding, stride):
        arr = np.array([[self.convolve2D(dataIn_i, kernel_i, padding, stride) for kernel_i in kernel] for dataIn_i in dataIn])
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
        return arr

    def gradient(self):
        return np.array([self.gradient2D()]*len(self.getPrevIn()))

    def gradient2D(self):
        arr = np.array([np.transpose(self.kernel, (0, 2, 1))]*len(self.getPrevIn()[0]))
        return arr.reshape(-1, arr.shape[-2], arr.shape[-1])

    def backward(self, gradIn):
        grad = self.gradient()
        return np.array([self.backward2D(grad_i, gradIn_i) for gradIn_i, grad_i in zip(gradIn, grad)])
        
    def backward2D(self, grad, gradIn):
        return np.array([self.convolve2D(np.pad(gradIn_i, self.kernel_size[0]-1, constant_values=0), grad_i) for gradIn_i, grad_i in zip(gradIn, grad)])

    def updateKernel(self, gradIn, epoch, learning_rate = 0.0001):
        for gradIn_i in gradIn:
            for dataIn_i in self.getPrevIn():
                self.updateKernel3D(gradIn_i, dataIn_i, epoch, learning_rate)

    def updateKernel3D(self, gradIn, dataIn, epoch, learning_rate = 0.0001):
        for i in range(len(dataIn)):
            for j in range(self.filters):
                self.updateKernel2D(gradIn[i + j], dataIn[i], epoch, learning_rate)

class PoolingLayer(Layer):
    def __init__(self, size, stride=1):
        super().__init__()
        self.size = size
        self.stride = stride

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(self.pool(dataIn))
        return self.getPrevOut()

    def pool(self, dataIn):
        return np.array([self.pool3D(dataIn[i]) for i in range(len(dataIn))])

    def pool3D(self, dataIn):
        return np.array([self.pool2D(dataIn[i]) for i in range(len(dataIn))])
    
    def pool2D(self, dataIn):
        dataHeight = dataIn.shape[0]
        dataWidth = dataIn.shape[1]

        outputWidth = int(((dataWidth - self.size) / self.stride) + 1)
        outputHeight = int(((dataHeight - self.size) / self.stride) + 1)
        output = np.zeros((outputHeight, outputWidth))

        for y in range(outputHeight):
            for x in range(outputWidth):
                output[y, x] = dataIn[y*self.stride:y*self.stride+self.size, x*self.stride:x*self.stride+self.size].max()

        return output
    
    def gradient(self):
        pass

    def backward(self, gradIn):
        return np.array([self.backward3D(grad_i, data) for data, grad_i in zip(self.getPrevIn(), gradIn)])
 
    def backward3D(self, gradIn, prevIn):
        return np.array([self.backwardRow(data, grad_i) for data, grad_i in zip(prevIn, gradIn)])

    def backwardRow(self, data, gradIn):
        dataHeight = data.shape[0]
        dataWidth = data.shape[1]
        output = np.zeros((dataHeight, dataWidth))

        for y in range(gradIn.shape[1]):
            for x in range(gradIn.shape[0]):
                grid = data[y*self.stride:y*self.stride+self.size, x*self.stride:x*self.stride+self.size]
                maxLoc = np.unravel_index(np.argmax(grid), (self.size, self.size))
                output[y*self.stride+maxLoc[0], x*self.stride+maxLoc[1]] = gradIn[y, x]

        return output

class FlattenLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward(self,dataIn):
        self.setPrevIn(dataIn)
        self.setPrevOut(self.flatten(dataIn))
        return self.getPrevOut()

    def flatten(self, dataIn):
        return np.array([dataIn[i].flatten() for i in range(len(dataIn))])
    
    def gradient(self):
        pass
 
    def backward(self, gradIn):
        return gradIn.reshape(self.getPrevIn().shape)
    
class DropoutLayer(Layer):
    def __init__(self, keep_prob):
        super().__init__()
        self.keep_prob = keep_prob

    def forward(self, dataIn, test=False, epoch=1):
        self.setPrevIn(dataIn)
        if (test):
            self.setPrevOut(dataIn)
            return dataIn
        else:
            np.random.seed(epoch)
            self.dropOutKey = np.random.rand(dataIn.shape[0], dataIn.shape[1]) < self.keep_prob
            dataOut = np.multiply(dataIn, self.dropOutKey)
            dataOut = dataOut / self.keep_prob
            self.setPrevOut(dataOut)
            return dataOut

    def gradient(self):
        tensor = np.ones_like(self.dropOutKey) / self.keep_prob
        return tensor

    def backward(self, gradIn):
        gradOut = gradIn * self.gradient()
        return gradOut
        
# Objective functions
class SquaredError():
    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  A single floating point value.
    def eval(self, Y, Yhat):
        #TODO
        return np.mean((Y - Yhat) ** 2)

    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  An N by K matrix.
    def gradient(self, Y, Yhat):
        #TODO
        return -2*(Y - Yhat)

class LogLoss():
    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  A single floating point value.
    def eval(self, Y, Yhat):
        #TODO
        return np.mean(-(Y * np.log(Yhat + EPSILON) + (1 - Y) * np.log(1 - Yhat + EPSILON)))

    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  An N by K matrix.
    def gradient(self, Y, Yhat):
        #TODO
        return -((Y - Yhat) / (Yhat * (1 - Yhat) + EPSILON))

class CrossEntropy():
    #Input: Y is an N by K matrix of target values.
    #Input: Yhat is an N by K matrix of estimated values.
    #Output:  A single floating point value.
    def eval(self, Y,Yhat):
        return -np.mean(np.sum(Y * np.log(Yhat + EPSILON), axis = 1))
    
    def gradient(self, Y, Yhat):
        return -np.divide(Y, (Yhat + EPSILON))
