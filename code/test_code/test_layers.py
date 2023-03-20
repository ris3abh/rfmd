import src.layers as layers
import numpy as np

convLayer = layers.Conv2DLayer(filters=1, kernel_size=(3,3), stride=1, padding=0)

X = np.array([[[1, 1, 0, 1, 0, 0, 1, 0],
               [1, 1, 1, 1, 0, 0, 1, 0],
               [0, 0, 1, 1, 0, 1, 0, 1],
               [1, 1, 1, 0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 0, 0, 1],
               [1, 0, 1, 0, 0, 1, 0, 1]],
              [[1, 1, 0, 1, 0, 0, 1, 0],
               [1, 1, 1, 1, 0, 0, 1, 0],
               [0, 0, 1, 1, 0, 1, 0, 1],
               [1, 1, 1, 0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1, 0, 1, 1],
               [0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 1, 1, 1, 0, 0, 1],
               [1, 0, 1, 0, 0, 1, 0, 1]]])
print(f'X.shape: {X.shape}')

kernel = np.array([[[2, -1, 2],
                   [2, -1, 0],
                   [1, 0, 2]],
                   [[2, -1, 2],
                   [2, -1, 0],
                   [1, 0, 2]]])

convLayer.setKernel(kernel)
result = convLayer.forward(X)
print("Convolution layer's result:\n", result)
print("Convolution layer's result shape:", result.shape)
expectedResult = np.array([[[[4, 7, 1, 7, 2, 1],
                            [6, 3, 5, 6, 4, 2],
                            [6, 5, 6, 4, 3, 7],
                            [4, 2, 5, 2, 5, 0],
                            [5, 6, 6, 2, 5, 3],
                            [2, 1, 2, 3, 2, 3]]]])
# assert np.array_equal(result, expectedResult)

poolingLayer = layers.PoolingLayer(3, 3)

result = poolingLayer.forward(result)
expectedResult = np.array([[[[7, 7], [6, 5]]]])
print("Pooling layer's result:", result)
print("Pooling layer's result shape:", result.shape)
# assert np.array_equal(result, expectedResult)

grad = np.array([[[[-2, 0],
                  [6, -2]],
                  [[-2, 0],
                  [6, -2]]],
                  [[[-2, 0],
                  [6, -2]],
                  [[-2, 0],
                  [6, -2]]]])
print("Gradient shape:", grad.shape)
poolingResult = poolingLayer.backward(grad)
expectedResult = np.array([[[[ 0, -2,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0, -2,  0],
                            [ 0,  6,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0]]]])
print("Pooling layer's gradient:\n", poolingResult)
print("Pooling layer's gradient shape:", poolingResult.shape)
# assert np.array_equal(poolingResult, expectedResult)

flattenLayer = layers.FlattenLayer()
print("Flatten layer's input shape:", result.shape)
result = flattenLayer.forward(result)
expectedResult = np.array([[7, 7, 6, 5]])
print("Flatten layer's result:", result)
print("Flatten layer's result shape:", result.shape)
# assert np.array_equal(result, expectedResult)

grad = np.array([[[-2, 0, 6, -2],[-2, 0, 6, -2]],[[-2, 0, 6, -2],[-2, 0, 6, -2]]])
print("Gradient shape:", grad.shape)
flattenResult = flattenLayer.backward(grad)
expectedResult = np.array([[[[-2, 0],
                            [6, -2]]]])
print("Flatten layer's gradient:", flattenResult)
print("Flatten layer's gradient shape:", flattenResult.shape)
# assert np.array_equal(flattenResult, expectedResult)

e = np.array([[[[0, -2, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, -2, 0],
               [0, 6, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]],
               [[0, -2, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, -2, 0],
               [0, 6, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]]])

kernel_T = np.array([[2, 2, 1],
                   [-1, -1, 0],
                   [2, 0, 2]])

print(f'Kernel backward: {np.transpose(kernel, (0, 2, 1))}')

result = convLayer.backward(poolingResult)
expectedResult = np.array([[[[ 0, -4,  0, -4,  0,  0,  0,  0],
                            [ 0,  0,  2,  2,  0,  0,  0,  0],
                            [ 0, -2, -4, -4,  0,  0,  0,  0],
                            [ 0,  0,  0,  0, -4,  0, -4,  0],
                            [ 0, 12,  0, 12,  0,  2,  2,  0],
                            [ 0,  0, -6, -6, -2, -4, -4,  0],
                            [ 0,  6, 12, 12,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0,  0,  0,  0]]]])
print("Convolution layer's result:\n", result)
# assert np.array_equal(result, expectedResult)

convLayer.updateKernel(e, 0)
print("Convolution layer's kernel:\n", convLayer.kernel)