import src.util as util
import src.layers as layers
import numpy as np
import os
from src.layers import *
import pandas as pd
import matplotlib.pyplot as plt

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

## reshaping the data 
X_train_img = X_train_img.reshape(X_train_img.shape[0], X_train_img.shape[1]*X_train_img.shape[2]*X_train_img.shape[3])
X_eval_img = X_eval_img.reshape(X_eval_img.shape[0], X_eval_img.shape[1]*X_eval_img.shape[2]*X_eval_img.shape[3])
X_test_img = X_test_img.reshape(X_test_img.shape[0], X_test_img.shape[1]*X_test_img.shape[2]*X_test_img.shape[3])

print("X_train_img.shape: ", X_train_img.shape)
print("Y_train.shape: ", Y_train.shape)
print("X_eval_img.shape: ", X_eval_img.shape)
print("Y_eval.shape: ", Y_test.shape)

## creating layers for the MLP model
L1 = InputLayer(X_train_img)
L2 = FullyConnectedLayer(X_train_img.shape[1], 1024)
L3 = ReluLayer()
Ld0 = DropoutLayer(0.4)
L4 = FullyConnectedLayer(1024, 512)
L5 = ReluLayer()
Ld1 = DropoutLayer(0.4)
L6 = FullyConnectedLayer(512, 256)
L7 = ReluLayer()
Ld2 = DropoutLayer(0.4)
L8 = FullyConnectedLayer(256, 64)
L9 = ReluLayer()
Ld3 = DropoutLayer(0.4)
L10 = FullyConnectedLayer(64, 4)
L11 = LogisticSigmoidLayer()
L12 = LogLoss()

Ld = [L1, L2, L3, Ld0, L4, L5, Ld1, L6, L7, Ld2, L8, L9, Ld3, L10, L11, L12]
L = [L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]

util.train_model(L, X_train_img, Y_train, X_eval_img, Y_eval, X_test_img, Y_test, "MLP_model", 
                 learning_rate = 0.0001, 
                 max_epochs = 5, 
                 batch_size = 8,
                 condition = 10e-10,
                 skip_first_layer=False)