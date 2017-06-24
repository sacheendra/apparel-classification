import numpy as np
from sklearn.ensemble import RandomForestClassifier

with open('fashion-data\\train-label.txt') as f1:
    line = f1.readlines()
line = [x.strip('\n') for x in line]
line = np.array(line[:499])
print line.shape
f1.close()

with open('fashion-data\\test-label.txt') as f1:
    actual_test_labels = f1.readlines()
actual_test_labels = [x.strip('\n') for x in actual_test_labels]
actual_test_labels = np.array(actual_test_labels)
f1.close()

container = np.load('fashion-data\surf0.npz')
data = [container[key].flatten() for key in container]
a = np.array(data)
container = np.load('fashion-data\surf1.npz')
data = [container[key].flatten() for key in container]
a = np.array(data)
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\surf2.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\hog0.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\hog1.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\hog2.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\lab0.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\lab1.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\lab2.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
# container = np.load('fashion-data\lbp.npz')
# data = [container[key].flatten() for key in container]
# b = np.array(data)
# a = np.concatenate((a, b), axis = 1)	#When using only LBP
train_data = np.concatenate((a, b), axis = 1)
print train_data.shape

container = np.load('fashion-data\surf0_test.npz')
data = [container[key].flatten() for key in container]
a = np.array(data)
container = np.load('fashion-data\surf1_test.npz')
data = [container[key].flatten() for key in container]
a = np.array(data)
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\surf2_test.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\hog0_test.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\hog1_test.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\hog2_test.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\lab0_test.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\lab1_test.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
a = np.concatenate((a, b), axis = 1)
container = np.load('fashion-data\lab2_test.npz')
data = [container[key].flatten() for key in container]
b = np.array(data)
# container = np.load('fashion-data\lbp_test.npz')
# data = [container[key].flatten() for key in container]
# b = np.array(data)
# a = np.concatenate((a, b), axis = 1) #When using only LBPs
test_data = np.concatenate((a, b), axis = 1)
print test_data.shape

score = 0
clf = RandomForestClassifier(n_jobs=2)
clf.fit(train_data, line)
print "Done with training the classifier"
result = clf.predict(test_data)
for i in range(len(result)):
	score += np.sum(actual_test_labels[i] == result[i])
accuracy = (score * 100)/(len(actual_test_labels))
print accuracy