import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist
import struct

def parse_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))

        if magic != 2051:
            raise ValueError("Invalid magic number in the MNIST image file")

        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_cols)

    return images

def parse_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))

        if magic != 2049:
            raise ValueError("Invalid magic number in the MNIST label file")

        labels = np.fromfile(f, dtype=np.uint8)

    return labels

def shift_scale_normalize(images):
    flat_images = images.reshape(images.shape[0], -1)
    min_val = np.min(flat_images)
    max_val = np.max(flat_images)
    images_normalized = (flat_images - min_val) / (max_val - min_val)

    images_normalized = images_normalized.reshape(images.shape)
    
    return images_normalized


def zero_mean_normalize(images):
    flat_images = images.reshape(images.shape[0], -1)

    # zero mean normalization formula is (x - mean) / std_dev
    images_normalized = (flat_images - np.mean(flat_images)) / np.std(flat_images)

    images_normalized = images_normalized.reshape(images.shape)
    
    return images_normalized


def compute_euclidean_distances(images):
    flat_images = images.reshape(images.shape[0], -1)
    num_images = flat_images.shape[0]
    image_length = flat_images.shape[1]

    distances = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(i, num_images):
            # loop through every pair of images
            # calculate the Euclidean distance between the two images
            
            dist = 0
            for k in range(image_length):
                dist += (flat_images[i][k] - flat_images[j][k])**2

            dist = np.sqrt(dist)
            
            distances[i, j] = dist
            distances[j, i] = dist

    return distances

def compute_cosine_similarity(image_set_1, image_set_2):
    # Flatten the images to 1D
    flat_images_1 = image_set_1.reshape(image_set_1.shape[0], -1)
    flat_images_2 = image_set_2.reshape(image_set_2.shape[0], -1)

    # Compute the dot product between the two sets of images
    dot_product = np.dot(flat_images_1, flat_images_2.T)
    
    # Calculate the norm of each image in both sets
    norm_1 = np.linalg.norm(flat_images_1, axis=1)
    norm_2 = np.linalg.norm(flat_images_2, axis=1)

    # Compute cosine similarity
    cosine_similarity = dot_product / np.outer(norm_1, norm_2)

    return cosine_similarity



def compute_euclidean_distances(image_set_1, image_set_2):
    distances = cdist(image_set_1.reshape(image_set_1.shape[0], -1), image_set_2.reshape(image_set_2.shape[0], -1), metric='euclidean')

    return distances



class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def predict_euclidean_distances(self, X_test):
        predictions = []
        distances = compute_euclidean_distances(X_test, self.X_train)

        for dist in distances:
            k_nearest_i = np.argsort(dist)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_nearest_i]

            predictions.append(max(set(k_nearest_labels), key=k_nearest_labels.count))
        
        return np.array(predictions)
    
    def predict_cosine_similarity(self, X_test):
        predictions = []
        cosine_similarities = compute_cosine_similarity(X_test, self.X_train)

        for sim in cosine_similarities:
            k_nearest_i = np.argsort(sim)[::-1][:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_nearest_i]
            
            predictions.append(max(set(k_nearest_labels), key=k_nearest_labels.count))

        return np.array(predictions)
    
    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)

images_path = 'train-images-idx3-ubyte'
labels_path = 'train-labels-idx1-ubyte'

images = parse_mnist_images(images_path)
labels = parse_mnist_labels(labels_path)
images = images[:1000]
labels = labels[:1000]

images = shift_scale_normalize(images)

split_train = int(len(images) * 0.8)
split_valid = int(len(images) * 0.9)

X_train, y_train = images[:split_train], labels[:split_train]
X_valid, y_valid = images[split_train:split_valid], labels[split_train:split_valid]
X_test, y_test = images[split_valid:], labels[split_valid:]

knn = KNNClassifier(k=3)
knn.fit(X_train, y_train)

y_valid_pred = knn.predict_euclidean_distances(X_valid)
validation_accuracy = knn.accuracy(y_valid, y_valid_pred)
print(f'Validation Accuracy: {validation_accuracy:.2f}')

y_test_pred = knn.predict_euclidean_distances(X_test)
test_accuracy = knn.accuracy(y_test, y_test_pred)
print(f'Test Accuracy: {test_accuracy:.2f}')
