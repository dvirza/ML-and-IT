import numpy as np
import random as rnd
from sklearn import datasets, linear_model, metrics

# Load diabetes dataset
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data  # matrix of dimensions 442x10

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

'''
# with scikit learn:
# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
mean_squared_error = metrics.mean_squared_error(diabetes_y_test, diabetes_y_pred)
print("Mean squared error: %.2f" % mean_squared_error)
print("="*80)
'''

# train
X = diabetes_X_train
y = diabetes_y_train

# train: init
W = np.random.rand(10) #weights vector
#W = Wl.reshape((10, 1))
b = rnd.random()
b_vec = (np.ones([422]))*b

learning_rate = 0.5
epochs = 6000



# train: gradient descent
for i in range(epochs):
    #calculate predictions
    Y_predictions = np.matmul(X, W)
    Y_predictions = Y_predictions + b_vec
    # calculate error and cost (mean squared error - use can use the imported function metrics.mean_squared_error)
    # TODO
#    print(np.shape(y,Y_predictions))
    mse = metrics.mean_squared_error(y,Y_predictions)
    print(mse)


    # calculate gradients
    # TODO
    grad = (1 / 422) * (np.matmul(np.transpose(X),Y_predictions - y))

    # update parameters
    # TODO
    W = W - learning_rate * grad

    b_vec = b_vec - (learning_rate/422)*(np.sum(Y_predictions-y))

    #Y_test_pred = np.matmul(diabetes_X_test, W) + b_vec

mse_test = metrics.mean_squared_error(diabetes_y_test,np.matmul(diabetes_X_test,W)+b_vec[:20])
print(f"MSE of test {mse_test}")
