import src.util as util
from sklearn.model_selection import train_test_split
import src.layers as layers
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def oneencoding(Yhat):
    Y_unique, Y_inverse = np.unique(Yhat, return_inverse=True)
    return np.eye(Y_unique.shape[0])[Y_inverse]

def load_rfmd_npy(file_loc):
    X = []
    for file in os.listdir(file_loc):
            img = np.load(file_loc+file)
            X.append(np.transpose(img))
    X = np.array(X)
    return X

row_size = 100
data_location = 'C:/Users/JS/Documents/Winter 22 Quarter/CS615/Project/rfmd'
X_train_img = load_rfmd_npy( data_location + '/data/Training_Set/preprocessed_numpy/')
X_train_img = X_train_img[0:row_size]

X_test_img = load_rfmd_npy(data_location + '/data/Evaluation_Set/preprocessed_eval/') #taking eval as a test right now
X_test_img = X_test_img[0:row_size]

Y_train = pd.read_csv( data_location + '/data/Training_Set/RFMiD_Training_Labels.csv')
Y_train = np.array(Y_train[['DR', 'MH', 'TSLN', 'ODC']]) #currently doing to for disease_risk - only binary classification
#Y_train = np.array(Y_train['Disease_Risk'])
Y_train = np.array(Y_train[0:row_size])
#Y_train = util.one_hot_array(Y_train, 4)

Y_test = pd.read_csv(data_location + '/data/Evaluation_Set/RFMiD_Validation_Labels.csv')
Y_test = np.array(Y_test[['DR', 'MH', 'TSLN', 'ODC']])#currently doing to for disease_risk - only binary classification
#Y_test = np.array(Y_test['Disease_Risk'])
Y_test = np.array(Y_test[0:row_size])
#Y_test = util.one_hot_array(Y_test, 4)

convLayer1 = layers.Conv3DLayer(filters=1, kernel_size=(3, 3), stride=1)
tanhLayer1 = layers.TanhLayer()
poolingLayer1 = layers.PoolingLayer(3, 3)
convLayer2 = layers.Conv3DLayer(filters=16, kernel_size=(2, 2), stride=1)
tanhLayer2 = layers.TanhLayer()
poolingLayer2 = layers.PoolingLayer(2, 2)
flattenLayer = layers.FlattenLayer()
fcLayer3 = layers.FullyConnectedLayer(2400, 120, xavier_init = True)
dropoutLayer3 = layers.DropoutLayer(0.9)
tanhLayer3 = layers.TanhLayer()
fcLayer4 = layers.FullyConnectedLayer(120, 84, xavier_init = True)
dropoutLayer4 = layers.DropoutLayer(0.8)
tanhLayer4 = layers.TanhLayer()
fcLayer5 = layers.FullyConnectedLayer(16428, 4, xavier_init = True)
softmaxLayer = layers.LogisticSigmoidLayer()
crossEntropyLoss = layers.CrossEntropy()
rfmd_layers = [convLayer1, tanhLayer1, poolingLayer1,
        convLayer2, tanhLayer2, poolingLayer2, flattenLayer, 
        fcLayer3, tanhLayer3, 
        fcLayer4, tanhLayer4, 
        fcLayer5, softmaxLayer, crossEntropyLoss]

rfmd_layers_test = [convLayer1, poolingLayer1, flattenLayer, fcLayer5, softmaxLayer, crossEntropyLoss]

util.train_model(rfmd_layers_test, X_train_img, Y_train, X_test_img, Y_test, "rmfd_model", 
                 learning_rate = 0.0001, 
                 max_epochs = 5, 
                 batch_size = 1,
                 condition = 10e-10,
                 skip_first_layer=False)