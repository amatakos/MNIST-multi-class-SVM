import numpy as np
import time
import matplotlib.pyplot as plt
import sys
from sklearn.svm import SVC
from mnist_loader import load_data_wrapper
from sklearn.svm.libsvm import predict_proba

#live printing on terminal
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)
sys.stdout = Unbuffered(sys.stdout)

#loading the dataset
(train_X, train_y), (test_X, test_y), xtest_original = load_data_wrapper()

print('Using the whole dataset with reduced dimensionality n=50')
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

#training
clf=SVC(C=100.0, kernel='poly', degree=3, gamma=0.1, probability = True)
start_time = time.time()
clf.fit(train_X,train_y)
print("Elapsed time for training: " + str(time.time() - start_time))
print('Parameters: ')
print("C = " + str(clf.get_params()["C"]))
print("degree = " + str(clf.get_params()['degree']))
print("gamma = " + str(clf.get_params()['gamma']))
print("kernel = " + str(clf.get_params()['kernel']))

#testing
plt.show()
k=3
y_pred = clf.predict(test_X)
y_proba = clf.predict_proba(test_X)
maxes = np.max(y_proba, axis=1)
min_indices = np.argpartition(maxes, k)
print('Probability of ' + str(k) + 'most difficult examples:')
print(maxes[min_indices[:k]])
print('Plotting the images of the most difficult examples')
for i in min_indices[:k]:
    data = np.reshape(xtest_original[i], (28,28))
    plt.imshow(data, cmap='gray')
    plt.show(block=True)
max_indices = np.argpartition(maxes, -k)
print('Probability of ' + str(k) + 'easiest examples:')
print(maxes[max_indices[:k]])
print('Plotting the images of the easiest examples')
for i in max_indices[:k]:
    data = np.reshape(xtest_original[i], (28,28))
    plt.imshow(data, cmap='gray')
    plt.show(block=True)
comp = np.array((y_pred == test_y), dtype=np.float)
print('Accuracy on training: ' + str(np.mean(comp)))
y_pred = clf.predict(train_X)
comp = np.array((y_pred == train_y), dtype=np.float)
print('Accuracy on test: ' + str(np.mean(comp)))
