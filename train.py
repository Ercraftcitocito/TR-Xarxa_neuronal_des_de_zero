import time

import numpy as np
import pandas as pd

# Carga y preparación del conjunto de entrenamiento
data_train = pd.read_csv('./MNIST/mnist_train.csv')
data_train = np.array(data_train)
np.random.shuffle(data_train)

data_train = data_train.T
Y_train = data_train[0]
X_train = data_train[1:] / 255.

# Carga y preparación del conjunto de test independiente
data_test = pd.read_csv('./MNIST/mnist_test.csv')
data_test = np.array(data_test)

data_test = data_test.T
Y_test = data_test[0]
X_test = data_test[1:] / 255.

# Inicialización de pesos y sesgos de la red neuronal
def init_params(neurons):
    W1 = np.random.rand(neurons, 784) - 0.5
    b1 = np.random.rand(neurons, 1) - 0.5
    W2 = np.random.rand(10, neurons) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z = Z - np.max(Z, axis=0, keepdims=True)
    A = np.exp(Z)
    A = A / np.sum(A, axis=0, keepdims=True)
    return A

# Propagación hacia adelante para obtener predicciones
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Propagación hacia atrás para calcular los gradientes
def backward_prop(Z1, A1, A2, W2, X, Y, m):
    one_hot_Y = one_hot(Y)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2

# Actualización de los parámetros usando la tasa de aprendizaje (alpha)
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Bucle principal de entrenamiento por épocas y mini-lotes
def gradient_descent_mini_batch(X, Y, alpha, epochs, batch_size):
    W1, b1, W2, b2 = init_params(128)  # Inicialización de pesos y sesgos
    m = X.shape[1]

    for epoch in range(epochs):
        # 1. Barajar los datos al inicio de cada época
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[permutation]

        # 2. Recorrer los mini-lotes
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[:, i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            current_batch_size = X_batch.shape[1]

            # 3. Propagación y actualización por lote
            Z1, A1, _, A2 = forward_prop(W1, b1, W2, b2, X_batch)
            dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X_batch, Y_batch, current_batch_size)
            W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        # 4. Mostrar métricas al finalizar cada época
        _, _, _, A2_full = forward_prop(W1, b1, W2, b2, X)
        predictions = get_predictions(A2_full)
        accuracy = get_accuracy(predictions, Y)
        print(f"Época {epoch + 1}/{epochs} | Precisión: {accuracy:.4f}")

    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Ejecución del entrenamiento y evaluación
print("Iniciando entrenamiento...")
start_time = time.time()
W1, b1, W2, b2 = gradient_descent_mini_batch(X_train, Y_train, alpha=0.1, epochs=20, batch_size=100)
end_time = time.time()
print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")

# Evaluación final con datos que la red nunca ha visto
predictions_test = make_predictions(X_test, W1, b1, W2, b2)
accuracy_test = get_accuracy(predictions_test, Y_test)
print(f"\nPrecisión final en conjunto de test: {accuracy_test:.4f} ({accuracy_test*100:.2f}%)")

# Guardar los pesos y sesgos entrenados
print("\nGuardando los parámetros del modelo...")
np.savez('pesos_modelo.npz', W1=W1, b1=b1, W2=W2, b2=b2)
print("Modelo guardado exitosamente en 'pesos_modelo.npz'.")
