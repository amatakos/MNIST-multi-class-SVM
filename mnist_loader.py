"""A library to load the MNIST image data."""
import pickle
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_data():
    """Return the MNIST data as a tuple containing the training data
    and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    ``test_data`` contains 10,000 images."""

    f = gzip.open("mnist.pkl.gz", 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, test_data)``.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``test_data`` is a list containing 10,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the test data.  These formats
    turn out to be the most convenient code."""
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    #test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]

    n_components = 784
    use_PCA = True
    if use_PCA:
        n_components = 50
        pca = PCA(n_components)

        x_train = np.hstack(training_inputs)
        pca.fit(x_train.transpose())
        x_transformed_train = pca.transform(x_train.transpose())
        training_inputs = x_transformed_train

        x_test = np.hstack(test_inputs)
        x_transformed_test = pca.transform(x_test.transpose())
        test_inputs = x_transformed_test

    training_data = (training_inputs, tr_d[1])
    test_data = (test_inputs, va_d[1])

    return training_data, test_data, x_test.transpose()
