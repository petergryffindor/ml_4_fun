from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
digits = datasets.load_digits()

### SVM
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[1:], digits.target[1:])
a = clf.predict([digits.data[70]])


print('end')
