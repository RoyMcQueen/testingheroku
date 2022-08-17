import pickle


def run_model(mylist): # the list will receive the independent variables (in this case, the sepal length and width and petal lngth and wdth)

    X_new = [mylist]

    with open('logistic_model.p', 'rb') as file:
        model = pickle.load(file)

    predictions = model.predict(X_new)

    if predictions == 0:
        name = 'Setosa'
    elif predictions == 1:
        name = 'Versicolor'
    elif predictions == 2:
        name = 'Virginica'

    else:
        name = ''
    
    return name

