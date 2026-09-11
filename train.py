import time

import numpy as np
import pandas as pd

# 1. Carrega i preparació de dades
print("Carregant MNIST...")
data_train = np.array(pd.read_csv('./MNIST/mnist_train.csv')).T
Y_train = data_train[0]
X_train = data_train[1:] / 255.0

data_test = np.array(pd.read_csv('./MNIST/mnist_test.csv')).T
Y_test = data_test[0]
X_test = data_test[1:] / 255.0

# 2. Inicialització He (Kaiming Normal) per a ReLU
def init_params(neurons):
    W1 = np.random.randn(neurons, 784) * np.sqrt(2.0 / 784)
    b1 = np.zeros((neurons, 1))
    W2 = np.random.randn(10, neurons) * np.sqrt(2.0 / neurons)
    b2 = np.zeros((10, 1))
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T

# 3. Propagació cap endavant
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# 4. Propagació cap enrere (Derivada simplificada de Cross-Entropy + Softmax)
def backward_prop(Z1, A1, A2, W2, X, Y):
    m = X.shape[1]
    dZ2 = A2 - one_hot(Y)
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * (Z1 > 0)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# 5. Entrenament per Mini-Lots (Mini-batch SGD)
def train(X, Y, lr=0.1, epochs=10, batch_size=50, neurons=256):
    W1, b1, W2, b2 = init_params(neurons)
    m = X.shape[1]

    for epoch in range(epochs):
        perm = np.random.permutation(m)
        X_sh, Y_sh = X[:, perm], Y[perm]

        for i in range(0, m, batch_size):
            X_b, Y_b = X_sh[:, i:i+batch_size], Y_sh[i:i+batch_size]

            Z1, A1, _, A2 = forward_prop(W1, b1, W2, b2, X_b)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X_b, Y_b)

            # Actualització (SGD)
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        # Precisió al final de l'època
        _, _, _, A2_eval = forward_prop(W1, b1, W2, b2, X)
        acc = np.sum(np.argmax(A2_eval, axis=0) == Y) / m * 100
        print(f"Època {epoch+1:02d}/{epochs:02d} | Precisió: {acc:.2f}%")

    return W1, b1, W2, b2

# 6. Entrenament
print("\nIniciant entrenament...")
start = time.time()
W1, b1, W2, b2 = train(X_train, Y_train, lr=0.1, epochs=15, batch_size=50, neurons=256)
print(f"Temps: {time.time() - start:.2f} segons")

# Test final
_, _, _, A2_test = forward_prop(W1, b1, W2, b2, X_test)
test_acc = np.sum(np.argmax(A2_test, axis=0) == Y_test) / Y_test.size * 100
print(f"Precisió en Test: {test_acc:.2f}%")

np.savez('weight&biases.npz', W1=W1, b1=b1, W2=W2, b2=b2)
print("Pesos desats a 'weight&biases.npz'.")
