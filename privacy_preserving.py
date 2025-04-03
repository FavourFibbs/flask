# privacy_preserving.py
import numpy as np

def add_laplace_noise(data, epsilon=0.1):
    noise = np.random.laplace(0, 1/epsilon, data.shape)
    return data + noise

def preprocess_data_with_privacy(features, epsilon=0.1):
    return add_laplace_noise(features, epsilon)
