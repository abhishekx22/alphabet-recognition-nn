"""
emnist_loader.py
~~~~~~~~~~~~~~~~~
 
A loader for the EMNIST 'letters' dataset, formatted to work with
network.py (based on Michael Nielsen's mnist_loader.py structure).
 
Uses torchvision's EMNIST dataset (already downloaded to ./emnist_data)
instead of the 'emnist' pip package, since torchvision's mirror is more
reliable to download.
 
EMNIST letters labels are 1-indexed (1 = 'A', 2 = 'B', ..., 26 = 'Z').
We keep that convention consistent across training, validation and
test data below.
"""
 
import numpy as np
from torchvision.datasets import EMNIST
 
 
def vectorized_result(j):
    """Return a 26-dimensional unit vector with a 1.0 in the j-1'th
    position and zeroes elsewhere. This is used to convert a letter's
    label (1...26) into a corresponding desired output from the
    neural network."""
    e = np.zeros((26, 1))
    e[j - 1] = 1.0
    return e
 
 
def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``.
 
    training_data: list of (784x1 input, 26x1 vectorized label) tuples
    validation_data / test_data: list of (784x1 input, int label-1) tuples
 
    The int labels in validation/test data are 0-indexed (0 = 'A',
    ..., 25 = 'Z') to match np.argmax() output from network.evaluate().
    """
    print("Loading EMNIST letters...")
 
    # download=False because the dataset is already downloaded to
    # ./emnist_data via the torchvision command run earlier.
    train_set = EMNIST(root='./emnist_data', split='letters',
                        train=True, download=False)
    test_set = EMNIST(root='./emnist_data', split='letters',
                       train=False, download=False)
 
    tr_images = train_set.data.numpy()   # shape (N, 28, 28), uint8
    tr_labels = train_set.targets.numpy()  # shape (N,), values 1..26
 
    te_images = test_set.data.numpy()
    te_labels = test_set.targets.numpy()
 
    training_data = [
        (img.flatten().reshape(784, 1) / 255.0, vectorized_result(int(label)))
        for img, label in zip(tr_images, tr_labels)
    ]
 
    test_data_full = [
        (img.flatten().reshape(784, 1) / 255.0, int(label) - 1)
        for img, label in zip(te_images, te_labels)
    ]
 
    n_val = int(len(test_data_full) * 0.5)
    validation_data = test_data_full[:n_val]
    test_data = test_data_full[n_val:]
 
    print(f"Training: {len(training_data)}, "
          f"Validation: {len(validation_data)}, "
          f"Test: {len(test_data)}")
 
    return training_data, validation_data, test_data
