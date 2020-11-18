import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm

# load the digits dataset
digits = datasets.load_digits()

# index to test-data element (excluded from training set)
idx = 0

# display digit
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[idx], cmap=plt.cm.gray_r, interpolation='nearest')
print (digits.images[idx])

# define index range
idx_range = np.r_[0:idx,idx+1:] if idx > 0 else slice(idx+1,len(digits.data))

# classify test-data element via SVM
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[idx_range], digits.target[idx_range])
a = clf.predict([digits.data[idx]])
print("Result: ", a)

plt.show()