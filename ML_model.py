# Machine Learning library and functions 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# pandas
import pandas as pd

# saving mode
import pickle

# loading the dataset
iris_bunch = load_iris() 

print(iris_bunch) # it's a bunch object which works like a dictionary ## we can use the debugging functionality to see what's inside iris bunch


iris_df = pd.DataFrame(iris_bunch['data'], columns = iris_bunch['feature_names']) # independent variables
print(iris_df.head(3))

target = iris_bunch['target'] # our dependent variable (0,1, or 2, one for each species of plant)

## split the train and the test

X_train, X_test, y_train, y_test = train_test_split(iris_df, target, random_state = 1, test_size = 0.2)


# let's create the model 

logistic = LogisticRegression(max_iter=1000)

# train the model

logistic.fit(X_train,y_train)

# predict
# print(logistic.predict(X_test))

# evaluate
# print(logistic.score(X_test,y_test))

## saving the model

pkl_file = 'logistic_model.p'

with open(pkl_file, 'wb') as file:
    pickle.dump(logistic, file)

