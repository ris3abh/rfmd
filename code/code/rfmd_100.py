import src.util as util
import src.layers as layers
import numpy as np
import os
import pandas as pd

np.set_printoptions(precision=5)

def oneencoding(Yhat):
    Y_unique, Y_inverse = np.unique(Yhat, return_inverse=True)
    return np.eye(Y_unique.shape[0])[Y_inverse]

def load_rfmd_npy(file_loc):
    X = []
    cwd = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cwd, file_loc)
    for file in os.listdir(path):
            img = np.load(file_loc+file)
            X.append(np.transpose(img))
    X = np.array(X)
    return X


row_size = 100

X_train_img = load_rfmd_npy('data/Training_Set/preprocessed_numpy_64/')
X_train_img = X_train_img[0:row_size]

X_eval_img = load_rfmd_npy('data/Evaluation_Set/preprocessed_eval_64/') #taking eval as a test right now
X_eval_img = X_eval_img[0:row_size]

X_test_img = load_rfmd_npy('data/Test_Set/preprocessed_test_64/') #taking eval as a test right now
X_test_img = X_test_img[0:row_size]


cwd = os.path.dirname(os.path.abspath(__file__))
Y_train = pd.read_csv(os.path.join(cwd, 'data/Training_Set/RFMiD_Training_Labels.csv'))
Y_train = np.array(Y_train[['DR', 'MH', 'TSLN', 'ODC']]) #currently doing to for disease_risk - only binary classification
Y_train = np.array(Y_train[0:row_size])

Y_eval = pd.read_csv(os.path.join(cwd, 'data/Evaluation_Set/RFMiD_Validation_Labels.csv'))
Y_eval = np.array(Y_eval[['DR', 'MH', 'TSLN', 'ODC']])#currently doing to for disease_risk - only binary classification
Y_eval = np.array(Y_eval[Y_eval.sum(axis=1)!= 0])
Y_eval = np.array(Y_eval[0:row_size])

Y_test = pd.read_csv(os.path.join(cwd, 'data/Test_Set/RFMiD_Testing_Labels.csv'))
Y_test = np.array(Y_test[['DR', 'MH', 'TSLN', 'ODC']])
Y_test = np.array(Y_test[Y_test.sum(axis=1)!= 0])
Y_test = np.array(Y_test[0:row_size])


#VGG16
#1 - filter_size = 64
convLayer1_1 = layers.Conv3DLayer(filters=8, kernel_size=(3, 3), stride=1) # working 32x32
reLuLayer1_1 = layers.ReluLayer() #working for 32x32
maxPooling1 = layers.PoolingLayer(size=2, stride = 2) # working 32x32
dropout1 = layers.DropoutLayer(0.25)

#3 - filter_size = 256
convLayer3_1 = layers.Conv3DLayer(filters=16, kernel_size=(3, 3), stride=1) # working 32x32
reLuLayer3_1 = layers.ReluLayer() # working 32x32
maxPooling2 = layers.PoolingLayer(size=2, stride = 2) # working 32x32
dropout2 = layers.DropoutLayer(0.5)

flattenLayer = layers.FlattenLayer()
fcLayer1 = layers.FullyConnectedLayer(75264, 400) #need to change 78643200 / 32  1228800 / 16
fcLayer2 = layers.FullyConnectedLayer(400, 4)
activationLayer = layers.LogisticSigmoidLayer()
crossEntropyLoss = layers.LogLoss()

rfmd_layers = [
                convLayer1_1, reLuLayer1_1, maxPooling1,
                convLayer3_1, reLuLayer3_1, maxPooling2,
               flattenLayer, fcLayer1, dropout2, fcLayer2, activationLayer,
               crossEntropyLoss
            ]


util.train_model(rfmd_layers, X_train_img, Y_train, X_eval_img, Y_eval, X_test_img, Y_test, "rmfd_model_4", 
                 learning_rate = 0.0001, 
                 max_epochs = 5, 
                 batch_size = 8,
                 condition = 10e-10,
                 skip_first_layer=False)