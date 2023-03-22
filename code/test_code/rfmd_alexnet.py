import src.util as util
import src.layers as layers
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=5)

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
data_location = 'C:/Git/rfmd'
X_train_img = load_rfmd_npy( data_location + '/data/Training_Set/preprocessed_numpy/')
X_train_img = X_train_img[0:row_size]
# X_train_img = X_train_img[:, :, :,np.newaxis]

X_test_img = load_rfmd_npy(data_location + '/data/Evaluation_Set/preprocessed_eval/') #taking eval as a test right now
X_test_img = X_test_img[0:row_size]
# X_test_img = X_test_img[:, :, :,np.newaxis]

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

#VGG16
#1 - filter_size = 64
convLayer1_1 = layers.Conv3DLayer(filters=8, kernel_size=(3, 3), stride=1) # working 32x32
reLuLayer1_1 = layers.ReluLayer() #working for 32x32
# convLayer1_2 = layers.Conv3DLayer(filters=8, kernel_size=(3, 3), stride=1)
# reLuLayer1_2 = layers.ReluLayer()
maxPooling1 = layers.PoolingLayer(size=2, stride = 2) # working 32x32
dropout1 = layers.DropoutLayer(0.25)
#2 - filter_size = 128
# convLayer2_1 = layers.Conv3DLayer(filters=128, kernel_size=(3, 3), stride=1)
# reLuLayer2_1 = layers.ReluLayer()
# convLayer2_2 = layers.Conv3DLayer(filters=128, kernel_size=(3, 3), stride=1)
# reLuLayer2_2 = layers.ReluLayer()
# maxPooling2 = layers.PoolingLayer(size=2, stride = 2)

#3 - filter_size = 256
convLayer3_1 = layers.Conv3DLayer(filters=16, kernel_size=(3, 3), stride=1) # working 32x32
reLuLayer3_1 = layers.ReluLayer() # working 32x32
# convLayer3_2 = layers.Conv3DLayer(filters=16, kernel_size=(3, 3), stride=1)
# reLuLayer3_2 = layers.ReluLayer()
# convLayer3_3 = layers.Conv3DLayer(filters=256, kernel_size=(3, 3), stride=1)
# reLuLayer3_3 = layers.ReluLayer()
maxPooling2 = layers.PoolingLayer(size=2, stride = 2) # working 32x32
dropout2 = layers.DropoutLayer(0.25)
#3 - filter_size = 512
# convLayer4_1 = layers.Conv3DLayer(filters=384, kernel_size=(3, 3), stride=1)
# reLuLayer4_1 = layers.ReluLayer()
# convLayer4_2 = layers.Conv3DLayer(filters=384, kernel_size=(3, 3), stride=1)
# reLuLayer4_2 = layers.ReluLayer()
# convLayer4_3 = layers.Conv3DLayer(filters=256, kernel_size=(3, 3), stride=1)
# reLuLayer4_3 = layers.ReluLayer()
# maxPooling3 = layers.PoolingLayer(size=2, stride = 2)

# gptConv1 = layers.Conv2DLayer(1, 6, 5)
# reLuLayer = layers.ReluLayer()
# gptConv2 = layers.Conv2DLayer(1, 16, 5)
# maxPooling1 = layers.MaxPooling2D()

flattenLayer = layers.FlattenLayer()
fcLayer1 = layers.FullyConnectedLayer(75264, 400) #need to change 78643200 / 32  1228800 / 16
fcLayer2 = layers.FullyConnectedLayer(400, 4)
activationLayer = layers.LogisticSigmoidLayer()
crossEntropyLoss = layers.LogLoss()

# rfmd_layers = [
#                 gptConv1, reLuLayer, maxPooling1,
#                 gptConv1, reLuLayer, maxPooling1,
#                 flattenLayer, fcLayer1, fcLayer2, activationLayer,
#                 crossEntropyLoss
#                 ]

rfmd_layers = [
                convLayer1_1, reLuLayer1_1, maxPooling1,
                convLayer3_1, reLuLayer3_1, maxPooling2,
               flattenLayer, fcLayer1, fcLayer2, activationLayer,
               crossEntropyLoss
            ]



            # convLayer1_2, reLuLayer1_2, maxPooling1,
# convLayer3_1, reLuLayer3_1, convLayer3_2, reLuLayer3_2, maxPooling2, dropout2,
# rfmd_layers = [convLayer1_1, reLuLayer1_1, maxPooling1,
#                convLayer4_3, reLuLayer4_3, maxPooling2,
#                flattenLayer, fcLayer1, fcLayer2, activationLayer,
#                crossEntropyLoss
#         ]


util.train_model(rfmd_layers, X_train_img, Y_train, X_test_img, Y_test, "rmfd_model", 
                 learning_rate = 0.001, 
                 max_epochs = 5, 
                 batch_size = 4,
                 condition = 10e-10,
                 skip_first_layer=False)