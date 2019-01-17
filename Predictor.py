# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#printing cor-realtion graph
"""
    sb.pairplot(dataset)

"""
#Check  NUll values
print("Total Number of Null Values are ")
print( dataset.isnull().sum())

#check out correlation between predictors and predictant
print("\nCor-relation matrix is \n")
print(dataset.corr())
# X is a matrix vector and y is a vector
# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

accuracy = regressor.score(X_train,y_train)
print("Accuracy of our model is ",100*accuracy)
counter =1
while counter:
    print("Enter your experience (in years) ")
    exp = int(input())
    print("Your Salary is ",regressor.predict(exp))
    print("Enter 0 to exit and 1 for continue ")
    counter = int(input())

