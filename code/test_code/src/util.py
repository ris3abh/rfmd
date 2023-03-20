import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import numpy as np
import src.layers as layers

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_training_data():
    cifar10_data = unpickle("cifar-10-batches-py/data_batch_1")
    cifar10_grey_images, cifar10_labels = convert_images_to_gray(cifar10_data)
    cifar10_grey_images /= 255 # normalize to [0,1]

    for i in range(2, 6, 1):
        cifar10_batch_data = unpickle(f"cifar-10-batches-py/data_batch_{i}")
        cifar10_batch_grey_images, cifar10_batch_labels = convert_images_to_gray(cifar10_batch_data)
        cifar10_batch_grey_images /= 255 # normalize to [0,1]
        cifar10_grey_images = np.concatenate((cifar10_grey_images, cifar10_batch_grey_images), axis=0)
        cifar10_labels = np.concatenate((cifar10_labels, cifar10_batch_labels), axis=0)

    return cifar10_grey_images, cifar10_labels

def load_test_data():
    cifar10_data = unpickle("cifar-10-batches-py/test_batch")
    cifar10_grey_images, cifar10_labels = convert_images_to_gray(cifar10_data)
    cifar10_grey_images /= 255 # normalize to [0,1]
    return cifar10_grey_images, cifar10_labels

def show_image(image, label, label_names):
    plt.imshow(image)
    plt.title(label_names[label].decode())
    plt.show()

def convert_images_to_gray(batch):
    converted_images = np.zeros((batch[b'data'].shape[0], 32, 32, 1))
    for i in range(len(batch[b'data'])):
        image = batch[b'data'][i]
        image = image.reshape(3,32,32)
        image = image.transpose(1,2,0)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image.reshape(32,32,1)
        converted_images[i] = image
    return converted_images, batch[b'labels']

def one_hot(y, n_classes):
    return np.eye(n_classes).astype(float)[y]

def one_hot_array(y, n_classes):
    return np.array([one_hot(y_i, n_classes) for y_i in y])

def decode(y_pred):
    return np.array([np.argmax(y_i) for y_i in y_pred]) 

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def forward(_layers, X, test=True, epoch=1):
    h = X
    for i in range(len(_layers)-1):
        if(isinstance(_layers[i], layers.DropoutLayer)):
            h = _layers[i].forward(h, test, epoch)
        else:
            h = _layers[i].forward(h)
    y_hat = h
    return y_hat

def train_model(layers_, X_train, Y_train, X_val, Y_val, filename="default", learning_rate = 0.001, max_epochs = 100, batch_size = 25, condition = 10e-10, skip_first_layer = True):
    epoch = 0
    lastEval = 0
    loss_train = []
    loss_val = []

    pbar1 = tqdm(total = max_epochs, desc='Model epochs', unit="epochs")
    num_batches = int(X_train.shape[0] / batch_size)
    pbar2 = tqdm(total = num_batches, desc='Model batches', unit="Batch")
    while (epoch < max_epochs):
        # shuffle data
        indices = np.random.permutation(X_train.shape[0])
        X_train = X_train[indices]
        Y_train = Y_train[indices]
        
        pbar2.refresh()
        pbar2.reset()
        for i in range(0, X_train.shape[0], batch_size):
            # get batch
            X_batch = X_train[i:i+batch_size]
            Y_batch = Y_train[i:i+batch_size]

            # perform forward propagation
            h = forward(layers_, X_batch, test=False, epoch=epoch)

            # perform backwards propagation, updating weights
            if skip_first_layer:
                start = 1
            else:
                start = 0
            grad = layers_[-1].gradient(Y_batch, h)
            for layer in reversed(layers_[start:-1]):
                newGrad = layer.backward(grad)
                if (isinstance(layer, layers.FullyConnectedLayer)):
                    layer.updateWeights(grad, epoch, learning_rate)
                if (isinstance(layer, layers.Conv2DLayer)) or (isinstance(layer, layers.Conv3DLayer)):
                    layer.updateKernel(grad, epoch, learning_rate)
                grad = newGrad
            pbar2.update(1)
            

        # evaluate loss for training
        Y_hat_train = forward(layers_, X_train)
        eval = layers_[-1].eval(Y_train, Y_hat_train)
        loss_train.append(eval)
        acc1 = model_Acc(Y_train, Y_hat_train, "Training")

        # finish training if change in loss is too small
        if (epoch > 2 and abs(eval - lastEval) < condition):
            break
        lastEval = eval

        # evaluate loss for validation
        Y_hat_val = forward(layers_, X_val)
        val_eval = layers_[-1].eval(Y_val, Y_hat_val)
        loss_val.append(val_eval)

        pbar1.set_description(f"Validation loss: {val_eval}")
        acc2 = model_Acc(Y_val, Y_hat_val, "Valuation")
        pbar1.set_description(f"Model Epochs (Train Acc: {format(acc1, '.4f')} Val Acc: {format(acc2, '.4f')}))")
        #print("Epoch: %d, Train Loss: %f, Val Loss: %f" % (epoch, eval, val_eval))
        
        epoch += 1
        pbar1.update(1)
    pbar2.close()
    pbar1.close()

    calculate_accuracy(X_train, Y_train, layers_, type = "Training")
    calculate_accuracy(X_val, Y_val, layers_, type = "Validation")
    

    # plot log loss
    plt.xlabel("Epoch")
    plt.ylabel("J")
    plt.plot(loss_train, label="Training Loss")
    plt.plot(loss_val, label="Validation Loss")
    plt.legend()
    plt.show()
    #plt.savefig(f'./lenet_figures/{filename}.png')
    plt.clf()

def model_Acc(y_true, y_pred, type = None, threshold=0.5):
    # convert y_pred to binary matrix
    y_pred_binary = (y_pred > threshold).astype(int)
    # calculate the element-wise comparison between y_true and y_pred
    comparison = (y_true == y_pred_binary)
    # calculate the mean accuracy across all samples and labels
    accuracy = np.mean(comparison)
    return accuracy

def calculate_accuracy(X, Y, layers, type = "Training"):
    Y_hat = forward(layers, X)
    accuracy = model_Acc(Y, Y_hat)
    #accuracy = (Y.argmax(axis=1) == Y_hat.argmax(axis=1)).mean() * 100
    print(f"{type} accuracy: {accuracy}")
    
def model_Acc1(y, y_hat, type = None):
    #Y_hat = np.argmax(Y_hat)
    #return (y == Y_hat).mean() * 100
    acc = 0
    for i in range(len(y)):
        y_argmax = np.argmax(y_hat[i], axis = 1)
        acc += y[i,y_argmax]
        
    acc = acc / len(y)
    # type tells us if we want to print or not
    if type is not None:
        print(f"{type} accuracy: {acc}")
        return None
    else:
        return acc