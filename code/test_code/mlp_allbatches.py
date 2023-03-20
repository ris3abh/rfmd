from cProfile import label
import sys
from time import sleep
from tqdm import tqdm
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import pickle
# from lenet import Y_test
from src.layers_2 import FullyConnectedLayer, InputLayer, LinearLayer, ReluLayer, LogisticSigmoidLayer, TanhLayer, SoftmaxLayer, DropoutLayer, SquaredError, LogLoss, CrossEntropy

def batch_to_image(data):
    X_train = data
    #print("Shape before reshape:", X_train.shape)
    # Reshape the whole image data
    X_train = X_train.reshape(len(X_train), 3, 32, 32)
    #print("Shape after reshape and before transpose:", X_train.shape)
    # Transpose the whole data
    X_train = X_train.transpose(0,2,3,1)
    #print("Shape after reshape and transpose:", X_train.shape)
    return X_train

def display_image(image_index, batch,  label_names):
    # take first image
    image = batch[b'data'][image_index]
    # take first image label index
    label = batch[b'labels'][image_index]
    # Reshape the image
    image = image.reshape(3,32,32)
    # Transpose the image
    image = image.transpose(1,2,0)
    # Display the image
    plt.imshow(image)
    plt.title(label_names[label].decode())

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data():

    x_train_list = []
    y_train_list = []
    batches_name = [r'cifar-10-batches-py/data_batch_'+str(i) for i in range(1,6,1)]
    for batch_name in batches_name:
        batch_data = unpickle(batch_name)
        x = batch_data[b'data']
        y = np.array(batch_data[b'labels'])
        x_train_list.append(x)
        y_train_list.append(y)
    test_batch = unpickle(r'cifar-10-batches-py/test_batch')
    x_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])

    return x_train_list, y_train_list, x_test, y_test

def oneHotEncode(labels):
    distict_outputs = 10
    oneHot = np.zeros((labels.size, distict_outputs))
    oneHot[np.arange(labels.size), labels] = 1
    return oneHot

def fowardProp(x, layers, test=True, epoch=1):
    #forwards!
    h = x
    for i in range(len(layers)-1):
        if(isinstance(layers[i], DropoutLayer)):
            h = layers[i].forward(h, test, epoch)
        else:
            h = layers[i].forward(h)
    y_hat = h
    return y_hat

def backwardProp(y_train, y_hat, layers, epochs, lr):
    grad = layers[-1].gradient(y_train, y_hat)
    for i in range(len(layers)-2,0,-1):
        new_grad = layers[i].backward(grad)
        if(isinstance(layers[i], FullyConnectedLayer)):
            layers[i].updateWeights(grad, epochs, lr)
        grad = new_grad

def test_split(data, frac= 2/3):
    two_thirds_rounded = math.ceil(frac * len(data))
    np.random.shuffle(data)
    train, test = data[:two_thirds_rounded,:], data[two_thirds_rounded:,:]
    return train, test

def trainModel(
    x_train_list, 
    y_train_list,
    x_test,
    y_test,
    layers, 
    epochs_lim = 120, 
    lr = .0001, 
    loss_lim = math.pow(10,-5), 
    batch_size = 2000
):


    # loss = 10000
    train_loss_ls = []
    test_loss_ls = []
    train_acc_ls = []
    test_acc_ls = []

    epoch = 1 
    loss_delta = 1
    pbar = tqdm(total = epochs_lim, desc='Training Model', unit="Epoch")
    while(epoch <= epochs_lim):
        if (loss_delta <loss_lim):
            break
            # pass
        #Shuffle
        for i in range(len(x_train_list)):
            X_train = x_train_list[i]
            Y_train = y_train_list[i]
            indices = np.random.permutation(X_train.shape[0])
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            # do batch training


            for i in range(0, X_train.shape[0], batch_size):
                # get batch
                X_batch = X_train[i:i+batch_size]
                Y_batch = Y_train[i:i+batch_size]

                # perform forward propagation
                Y_batch_hat = fowardProp(X_batch, layers, test=False, epoch=epoch)
                # perform backward propagation
                backwardProp(Y_batch, Y_batch_hat, layers, epoch, lr)
            
        x_train = x_train_list[0]
        y_train = y_train_list[0]
        y_train_hat = fowardProp(x_train, layers)
        train_loss = layers[-1].eval(y_train, y_train_hat)
        train_loss_ls.append(train_loss)
        train_acc = model_Acc(y_train, y_train_hat)
        train_acc_ls.append(train_acc)

        
        y_test_hat = fowardProp(x_test, layers)

        test_loss = layers[-1].eval(y_test, y_test_hat)
        test_loss_ls.append(test_loss)
        test_acc = model_Acc(y_test, y_test_hat, type = "Testing")
        test_acc_ls.append(test_acc)
     

        epoch += 1
        pbar.update(1)
    pbar.close()

    # # Final Accuracies
    # model_Acc(y_train, y_train_hat,type = "Training")
    # model_Acc(y_test, y_test_hat, type = "Testing")
    return layers, train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls


def model_Acc(y, y_hat, type = "Training"):
    # y_hat = fowardProp(x, layers)
    acc = 0

    for i in range(len(y)):
        acc += y[i,np.argmax(y_hat[i])]

    acc = acc / len(y)
    # print(f"{type} accuracy: {acc}")
    return acc

def plotMetrics(train_loss, test_loss, train_acc, test_acc, fig_name):
    fig, axs = plt.subplots(2, 1,  figsize=(6, 8))
    axs[0].plot(train_loss, label="Train" )
    axs[0].plot(test_loss,label="Test" )
    axs[0].set_ylabel('Cross Entropy')
    axs[1].plot(train_acc, label = "Train")
    axs[1].plot(test_acc, label = "Test")
    axs[1].set_ylabel('Accuracy')
    
    for ax in axs.flat:
        ax.set(xlabel='Epoch')
    for ax in axs.flat:
        ax.label_outer()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc= 'center')
    plt.savefig(r'MLP_figures/'+fig_name+".png")

def plotMetrics_dict(dict_, hp = 'LR'):
    hp_ls = list(dict_.keys())
    fig, axs = plt.subplots(2, 3,  figsize=(15, 10))

    # layers, train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls
    for i in range(len(hp_ls)):
        axs[0,i].plot(dict_[hp_ls[i]][0], label="Train" )
        axs[0,i].plot(dict_[hp_ls[i]][1],label="Test" )
        axs[0,i].set_title(hp + '=' + str(hp_ls[i]))
    
    for i in range(len(hp_ls)):
        axs[1,i].plot(dict_[hp_ls[i]][2], label="Train" )
        axs[1, i].plot(dict_[hp_ls[i]][3],label="Test" )
        # axs[i,0].set_title(hp + '=' + str(hp_ls[i]))
    
    for ax in axs.flat[:3]:
        ax.set(xlabel='Epoch', ylabel='Cross Entropy Loss')
        ax.set_ylim(1,2.5)
    
    for ax in axs.flat[3:]:
        ax.set(xlabel='Epoch', ylabel='Accuracy')
        ax.set_ylim(0.3,0.6)

    for ax in axs.flat:
        ax.label_outer()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc= 'lower center')

    plt.savefig(r'MLP_figures/'+hp+".png")
    plt.show()


def MLP(epochs_lim, layer_1_size , layer_2_size, model_name = 'MLP_Xavier001_50ep'):
    print(model_name)
    x_train_list, y_train_list, x_test, y_test = load_data()
    y_train_list_OHE = [oneHotEncode(y) for y in y_train_list]
    y_test_OHE = oneHotEncode(y_test)


    L1 = InputLayer(x_train_list[0])
    L2 = FullyConnectedLayer(x_train_list[0].shape[1], layer_1_size, xavier_init=True, Adam=True)
    L3 = ReluLayer()
    #L4 = DropoutLayer(0.8)
    L5 = FullyConnectedLayer(layer_1_size, layer_2_size, xavier_init=True, Adam=True)
    L6 = ReluLayer()
    L7 = DropoutLayer(0.6)
    L8 = FullyConnectedLayer(layer_2_size, 10, xavier_init=True, Adam=True)
    L9 = SoftmaxLayer()
    L10 = CrossEntropy()

    layers = [L1, L2, L3, L5, L6,L7, L8, L9, L10]

    layers, train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls = trainModel(
        x_train_list, 
        y_train_list_OHE,
        x_test,
        y_test_OHE,
        layers, 
        epochs_lim, 
        lr = 0.0001, 
        loss_lim=-1, 
        batch_size=2000)
    model_result = [train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls]

    with open(model_name, 'wb') as fp:
        pickle.dump(model_result, fp)
    print(f"Final Training Accuracy: {train_acc_ls[-1]}")
    print(f"Final Testing Accuracy: {test_acc_ls[-1]}")

    plotMetrics(train_loss_ls, test_loss_ls, train_acc_ls, test_acc_ls, fig_name = model_name)

   

if __name__ == '__main__':
    MLP(50, layer_1_size = 1024, layer_2_size = 96) 

