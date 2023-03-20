import src.util as util
from sklearn.model_selection import train_test_split
import src.layers as layers
import numpy as np

cifar10_grey_images, cifar10_labels = util.load_training_data()
cifa10_test_grey_images, cifa10_test_labels = util.load_test_data()

X_train = cifar10_grey_images.squeeze()
Y_train = cifar10_labels
X_test = cifa10_test_grey_images.squeeze()
Y_test = cifa10_test_labels

Y_train_encoded = util.one_hot_array(Y_train, 10)
Y_test_encoded = util.one_hot_array(Y_test, 10)

convLayer1 = layers.Conv2DLayer(filters=6, kernel_size=(5, 5), stride=1)
tanhLayer1 = layers.TanhLayer()
poolingLayer1 = layers.PoolingLayer(2, 2)
convLayer2 = layers.Conv3DLayer(filters=16, kernel_size=(5, 5), stride=1)
tanhLayer2 = layers.TanhLayer()
poolingLayer2 = layers.PoolingLayer(2, 2)
flattenLayer = layers.FlattenLayer()
fcLayer3 = layers.FullyConnectedLayer(2400, 120, xavier_init = True)
dropoutLayer3 = layers.DropoutLayer(0.9)
tanhLayer3 = layers.TanhLayer()
fcLayer4 = layers.FullyConnectedLayer(120, 84, xavier_init = True)
dropoutLayer4 = layers.DropoutLayer(0.8)
tanhLayer4 = layers.TanhLayer()
fcLayer5 = layers.FullyConnectedLayer(84, 10, xavier_init = True)
softmaxLayer = layers.SoftmaxLayer()
crossEntropyLoss = layers.CrossEntropy()
lenet = [convLayer1, tanhLayer1, poolingLayer1,
        convLayer2, tanhLayer2, poolingLayer2, flattenLayer, 
        fcLayer3, tanhLayer3, 
        fcLayer4, tanhLayer4, 
        fcLayer5, softmaxLayer, crossEntropyLoss]

util.train_model(lenet, X_train, Y_train_encoded, X_test, Y_test_encoded, "lenet_50000_defaultmodel", 
                 learning_rate = 0.0001, 
                 max_epochs = 5, 
                 batch_size = 1,
                 condition = 10e-10,
                 skip_first_layer=False)
