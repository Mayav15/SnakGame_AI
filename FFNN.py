import numpy as np

n_x = 7                             # INPUT NODES
n_h = 9                             # NODES IN 1st HIDDEN LAYER
n_h2 = 15                           # NODES IN 2nd HIDDEN LAYER
n_y = 3                             # OUTPUT NODES
W1_shape = (9,7)                    
W2_shape = (15,9)                   
W3_shape = (3,15)                   

def get_weights_from_encoded(individual):

    expected_length = W1_shape[0] * W1_shape[1] + W2_shape[0] * W2_shape[1] + W3_shape[0] * W3_shape[1]
    
    if len(individual) != expected_length:
        raise ValueError(f"Incorrect length of individual: Expected {expected_length}, got {len(individual)}")

    # Weights between input and hidden layer 1
    W1 = individual[0:W1_shape[0] * W1_shape[1]]
    # Weights between the two hidden layers
    W2 = individual[W1_shape[0] * W1_shape[1]:W2_shape[0] * W2_shape[1] + W1_shape[0] * W1_shape[1]]
    # Weights between hidden layer 2 and output
    W3 = individual[W2_shape[0] * W2_shape[1] + W1_shape[0] * W1_shape[1]:]

    return (W1.reshape(W1_shape[0], W1_shape[1]), W2.reshape(W2_shape[0], W2_shape[1]), W3.reshape(W3_shape[0], W3_shape[1]))


def softmax(z):
    if np.any(np.isnan(z)):
        raise ValueError(f"Softmax input contains NaN: {z}")
    s = np.exp(z.T) / np.sum(np.exp(z.T), axis=1).reshape(-1, 1)
    if np.any(np.isnan(s)):
        raise ValueError(f"Softmax output contains NaN: {s}")
    return s

def forward_prop(X, individual):
    W1, W2, W3 = get_weights_from_encoded(individual)

    Z1 = np.matmul(W1, X.T)
    A1 = np.tanh(Z1)
    Z2 = np.matmul(W2, A1)
    A2 = np.tanh(Z2)
    Z3 = np.matmul(W3, A2)
    A3 = softmax(Z3)

    return A3