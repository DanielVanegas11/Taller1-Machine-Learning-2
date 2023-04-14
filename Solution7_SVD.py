from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Get Mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# build masks 
masktrain = (y_train == 0) | (y_train == 8)
masktest = (y_test == 0) | (y_test == 8)

# Apply masks
X_train = X_train[masktrain]
y_train = y_train[masktrain]
X_test = X_test[masktest]
y_test = y_test[masktest]

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]* X_train.shape[2])   
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]* X_test.shape[2])  

# Perform SVD on the training data
svd = TruncatedSVD(n_components=2)
reduced_x_train = svd.fit_transform(X_train)

# Plot the reduced data
plt.scatter(reduced_x_train[:,0], reduced_x_train[:,1], c=y_train)
plt.show()

# Perform SVD on the test data
reduced_x_test = svd.transform(X_test)

# Build the logistic regression model
regressor = LogisticRegression(random_state=0)
regressor.fit(reduced_x_train, y_train)
y_pred = regressor.predict(reduced_x_test)

# Evaluate the model performance on the test data
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy: ', accuracy_score(y_test, y_pred))
print('The performance using the sklearn library compared to the manual implementation is quite similar although it shows a slight improvement. However, it is evident that the performance is not 100% due to the loss of information when dimensionality reduction is performed."')
