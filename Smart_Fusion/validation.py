from sklearn.model_selection import KFold
def kfold(x,y):
    '''
    Inputs -- x - independent variables
              y -  dependent variables
    Outputs -- Predicted values of dependent variable y
               and the accuracy score of the model.
    Description -- This function takes the independent and 
                dependent variables and the Kfold validation is
                applied to get training and test set.
    '''
    kf = KFold(n_splits=10)
    kf.get_n_splits(x)
    for train_index,test_index in kf.split(x):
        x_train , x_test = x[train_index] , x[test_index]
        y_train , y_test = y[train_index] , y[test_index]
    return x_train , x_test , y_train, y_test