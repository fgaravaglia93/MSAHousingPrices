''' Francesco Garavaglia 889105, Metodi Statistici per l'apprendimento 2019/2020'''
''' Experimental Project "Housing Prices" '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from pandas import DataFrame
from os import listdir
from os.path import isfile, join


''' initialization  -----------------------------------------------------------'''

house_pricing_path = './dataset/'
house_pricing_cols = ['longitude', 'latitude', 'housing_median_age',
                     'total_rooms', 'total_bedrooms', 'population', 'households',
                     'median_income', 'median_house_value', 'ocean_proximity']
categoric_attr = 'ocean_proximity'
missing_attr = 'total_bedrooms'
related_attr = 'total_rooms'
target = 'median_house_value'
categoric_val = ['NEAR BAY', 'NEAR OCEAN', '<1H OCEAN', 'INLAND']

#csv extraction
csv_file = [f for f in listdir(house_pricing_path) if isfile(join(house_pricing_path, f))]
csv_file = csv_file[0]
house_price = pd.read_csv(house_pricing_path + csv_file, sep=';')



''' PRE PROCESSING ON DATASET '''

''' encode the categorical attribute into numeric attribute. 
Order is chosen by the average target value of each categoric value (desc)'''
def orderedIntegerEncoder(data, categoric_val, categoric_attr, target):
    
    #init variables
    data_enc = data.copy()  
    num_labels = len(categoric_val)
    avg_array = np.zeros(num_labels)
    count_categ = np.zeros(num_labels)
    sum_array = np.zeros(num_labels)
    
    #calculate average target value for each eligible string for categoric_attr
    for x in range(data.shape[0]):
        for i in range(num_labels):
            if data[categoric_attr].values[x] == categoric_val[i]: 
                sum_array[i] += data[target].values[x]
                count_categ[i] += 1
                break
    for i in range(num_labels):          
        avg_array[i] = sum_array[i]/count_categ[i]
    
    #set an order for the categoric attribute  values
    avg_series = pd.DataFrame({ 'avg': avg_array, 'categoric': categoric_val })
    avg_series = avg_series.sort_values('avg')
    #order 1 means lower median price of the house
    avg_series['order'] = pd.DataFrame([4,3,2,1])
    
    #apply the encoding on the dataset
    encoded_array = np.zeros(data.shape[0])
    
    for x in range(data.shape[0]):
        for i in range(num_labels):
                if data[categoric_attr].values[x] == categoric_val[i]: 
                    encoded_array[x] = avg_series['order'][i]
                    break
    data_enc['ocean_proximity_encoded'] = pd.DataFrame(encoded_array)
    data_enc.drop([categoric_attr], 1, inplace=True)
    return data_enc, avg_series

''' encode the categorical attribute into binary features (NOT USED)'''
def oneHotEncoder(data, categoric_val, categoric_attr):    
    data_enc = data.copy()    
    for i in range(len(categoric_val)):
        categoric_val_col = np.zeros(data.shape[0])
        #assign zero
        for x in range(data.shape[0]):
            if categoric_val[i] == data[categoric_attr].values[x]:            
                categoric_val_col[x] = 1
        data_enc[categoric_val[i]]=pd.DataFrame(categoric_val_col)

    data_enc.drop([categoric_attr], 1, inplace=True)
    return data_enc


''' imputation for missing values '''
def imputation(data, missing_attr, related_attr):
    # div  = #rooms / #bedrooms
    count = 0
    missing_index_array = []
    sum_rooms_div_bedrooms = 0
    
    '''find missing values location'''
    for x in range(data.shape[0]):
        
        if np.isnan(data[missing_attr].values[x]):
            missing_index_array.append(x)
            count+=1
        else:
            rooms =  data[related_attr].values[x] 
            rooms_div_bedrooms = rooms/ data[missing_attr].values[x]
            sum_rooms_div_bedrooms += rooms_div_bedrooms
            
    percentage = count/data.shape[0]*100
    #print('% of missing values of the attribute: {0}'.format(percentage))
    '''average div of number of number of rooms and bedrooms''' 
    avg_rooms_div_bedrooms =  (sum_rooms_div_bedrooms / (data.shape[0] - count))
    #print(avg_rooms_div_bedrooms)
    estimated_bedrooms_values  = pd.DataFrame({ missing_attr: [],related_attr: []})
    
    '''assign estimated values to the missing values, basing on related #total rooms attribute'''
    for y in range(len(missing_index_array)):
        rooms_value =  data[related_attr].values[missing_index_array[y]]
        
        data[missing_attr].values[missing_index_array[y]] =  rooms_value / avg_rooms_div_bedrooms
        tempDf = pd.DataFrame({ missing_attr : [data[missing_attr].values[missing_index_array[y]]], 
                               related_attr : [data[related_attr].values[missing_index_array[y]]]})
        estimated_bedrooms_values = pd.concat([estimated_bedrooms_values, tempDf])
                                                        
    return estimated_bedrooms_values




''' NORMALIZATION '''

''' extract min and max values for each feature column of a dataset '''
def datasetMinMax(data):
    min_max = []
    for i in range(len(data.values[0])):
        col_values = [row[i] for row in data.values]
        value_min = min(col_values)
        value_max = max(col_values)
        min_max.append([value_min, value_max])
    return min_max
 
''' rescale data on a given range (default 0-1)'''
def rescaleDataset(data, min_max, lower_bound=0, upper_bound=1):
    for row in data.values:
        for i in range(len(row)):
            #min_max[i][0] min value, min_max[i][0] max value for the column
            row[i] = (upper_bound - lower_bound)*(row[i] - min_max[i][0])/(min_max[i][1] - min_max[i][0])+lower_bound


''' CROSS VALIDATION '''

''' perform cross validation to find best prediction model given hyperparameter lambda set'''
def crossValidationRidge(X,Y,k=10):
    
    '''split in train and test set'''
    rate = 0.75
    n_samples = X.shape[0]
    n_samples_train = int(n_samples*rate)
    n_samples_valid = n_samples - n_samples_train
    
    X_train = X[0:n_samples_train]
    Y_train = Y[0:n_samples_train]
    
    X_test = X[n_samples_train:n_samples]
    Y_test = Y[n_samples_train:n_samples]   
    
    n_lambda = 11
    ridge_lambda_values = np.linspace(0,1,n_lambda)
    folds = int(n_samples_train/k)

    min_error = 1
    best_lambda = 0

    for x in range(len(ridge_lambda_values)):
        '''ridge regression lambda costraint on theta'''
        rmse_statistic_risk = []
        maape_statistic_risk = []
        ridge_lambda = ridge_lambda_values[x]
        print('Ridge Lambda hyper parameter: {0}'.format(ridge_lambda))   
        
        for i in range(k):
            
            X_train_fold, Y_train_fold, X_valid_fold, Y_valid_fold = splitKFolds(X_train, Y_train, i, folds, k)
        
            theta, cost =ridge_regression(X_train_fold, Y_train_fold, ridge_lambda)
            rmse, maape, pred = predictRidge(X_valid_fold, Y_valid_fold, theta)
            rmse_statistic_risk.append(rmse)
            maape_statistic_risk.append(maape)
            print('CV K FOLD {0} rmse {1}'.format(i, rmse))

        avg_rmse = sum(rmse_statistic_risk)/len(rmse_statistic_risk)
        avg_maape = sum(maape_statistic_risk)/len(maape_statistic_risk)
        print('\naverage statistic risk for ridge lambda factor  = {0}: RMSE - {1} MAAPE - {2}'.format(ridge_lambda, avg_rmse, avg_maape))
        if avg_rmse < min_error:
            min_error = avg_rmse
            best_lambda = ridge_lambda
     
    
    #test the best model
          
    theta, cost =ridge_regression(X_train, Y_train, best_lambda)
    rmse, maape, pred = predictRidge(X_test, Y_test, theta, plot=True)
    pred = pd.DataFrame(pred)      
    min_max = datasetMinMax(pred)
    rescaleDataset(pred, min_max,min_max_target[0],min_max_target[1]) 
    print('Best model: Ridge Lambda hyper parameter: {0}'.format(best_lambda))
    print('RMSE - {0} \nMAAPE - {1}'.format(rmse,maape))
    
    return pred

''' perform nested cross validation to find best prediction model '''
def nestedCrossValidationRidge(X,Y,outer_k=10, inner_k=3):
    
    #split in train and test set    
    rate = 0.75
    n_samples = X.shape[0]
    n_samples_train = int(n_samples*rate)
    n_samples_valid = n_samples - n_samples_train
    
    X_train = X[0:n_samples_train]
    Y_train = Y[0:n_samples_train]
    
    X_test = X[n_samples_train:n_samples]
    Y_test = Y[n_samples_train:n_samples]   
    
    n_lambda = 11
    ridge_lambda_values = np.linspace(0,1,n_lambda)
    folds = int(n_samples_train/outer_k)
    
    ''' k-fold outer cross validation '''
    min_rmse_out = 1
    
    
    ''' ridge regression lambda costraint on theta '''
    rmse_statistic_risk = []
    maape_statistic_risk = []
    rmse_inner_statistic_risk = []
    
    ''' outer loop '''
    for i in range(outer_k):
        print('K FOLD {0}\n'.format(i)) 
        X_train_fold, Y_train_fold, X_valid_fold, Y_valid_fold = splitKFolds(X_train, Y_train, i, folds, outer_k)
            
        ''' inner loop '''
        inner_folds = int(X_train_fold.shape[0]/inner_k)
        
        min_rmse_in = 1
        for x in range(len(ridge_lambda_values)):
            
            ridge_lambda = ridge_lambda_values[x]
            sum_rmse = 0
            for j in range(inner_k):
                
                X_train_infold, Y_train_infold, X_valid_infold, Y_valid_infold = splitKFolds(X_train_fold, Y_train_fold, j, inner_folds, inner_k)    
                theta, cost = ridge_regression(X_train_infold, Y_train_infold, ridge_lambda)
                rmse, maape, pred = predictRidge(X_valid_infold, Y_valid_infold, theta)
                print('INNER CV. Ridge Lambda hyper parameter: {0} - rmse {1}'.format(ridge_lambda, rmse))
                sum_rmse += rmse
                
            ''' avg rmse for same lambda on inner cross validation k = inner_k '''
            avg_rmse_lambda = sum_rmse/inner_k 
            rmse_inner_statistic_risk.append(avg_rmse_lambda)
              
            
            if avg_rmse_lambda < min_rmse_in:
                min_rmse_in = avg_rmse_lambda
                best_lambda_in = ridge_lambda 
            
            print('Inner CV, Ridge Lambda hyper parameter: {0} - avg rmse {1}\n'.format(ridge_lambda, avg_rmse_lambda))                                
                
        ''' outer CV '''
        ''' best lambda was the one chosen from inner cv for that outer k fold '''
        theta, cost = ridge_regression(X_train_fold, Y_train_fold, best_lambda_in)
        rmse, maape, pred = predictRidge(X_valid_fold, Y_valid_fold, theta)        
        rmse_statistic_risk.append(rmse)
        maape_statistic_risk.append(maape)
        
        if rmse < min_rmse_out:
            min_rmse_out = rmse
            best_lambda_out = best_lambda_in
            
        
        print('\n OUTER CV  step K FOLD {0} Best Ridge Lambda hyper parameter: {1} - min avg rmse {2}\n'.format(i,best_lambda_in, min_rmse_in))  
        print('---------------------------------------------------------------------------------')

    avg_rmse_out = sum(rmse_statistic_risk)/len(rmse_statistic_risk)
    avg_maape_out = sum(maape_statistic_risk)/len(maape_statistic_risk)
    print('average statistic risk for ridge (RMSE)  = {0}: RMSE - {1} MAAPE - best lambda found: {2}'.format(avg_rmse_out, avg_maape_out, best_lambda_out))
       
    ''' test the best model (best_lambda_out) '''
    theta, cost =ridge_regression(X_train, Y_train, best_lambda_out)
    rmse, maape, pred = predictRidge(X_test, Y_test, theta, plot=True)
    print('Best model: Ridge Lambda hyper parameter: {0}'.format(best_lambda_out))
    print('RMSE - {0} \nMAAPE - {1}'.format(rmse,maape)) 
    return pred


''' split training set into k folds of same dimension '''
def splitKFolds(X_train, Y_train, i, folds, k):
    
    X_train_fold = X_train[0*i:i*folds]
    Y_train_fold = Y_train[0*i:i*folds]
        
    if i< k-1:
        X_train_fold = pd.concat([X_train_fold, X_train[(i+1)*folds:]])
        Y_train_fold = pd.concat([Y_train_fold, Y_train[(i+1)*folds:]])
                                  
    X_valid_fold = X_train[i*folds:(i+1)*folds]
    Y_valid_fold = Y_train[i*folds:(i+1)*folds]
    
    return X_train_fold, Y_train_fold, X_valid_fold, Y_valid_fold




''' RIDGE REGRESSION '''
''' apply ridge regression with gradient descent to minimize cost function '''
def ridge_regression(X_train, Y_train, ridge_lambda, plot=False):
    
    '''learning rate'''
    alpha = 0.05
    
    iterations = 6000
    
    
    '''weights are setted to one at the first iteration'''
    theta = np.ones(X_train.shape[1])
    
    '''minimize cost function with gradient descend'''
    theta, cost = gradientDescent(X_train.values, Y_train.values, theta, alpha, iterations, ridge_lambda)
    
   
    '''plot convergence'''
    if plot==True:
        print(theta)
        plt.figure(figsize=(7,6), dpi=100)
        plt.title('Ridge Regression \n alpha = {0}, iter = {1}, lambda = {2}\n'.format(alpha,iterations,ridge_lambda))
        plt.plot(np.arange(iterations), cost, 'b')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.yscale('log')
        
    return (theta, cost)

''' apply gradient descent to minimize a cost function '''
'''ridge_lambda = 0 --> linear regression'''    
def gradientDescent(X_train, Y_train, theta, alpha=0.05, iterations=6000, ridge_lambda=0):
    
    Y_train = Y_train[:,0]
    '''cost is used to plot convergence of the model'''
    cost = np.zeros(iterations)
    for i in range(iterations):
        
        ''' theta - learning rate * X*J. J = 1/N sum_N error^2'''
        predictions = np.dot(X_train, theta)
        loss = predictions - Y_train
        
        ''' ridge_lambda * theta is used for ridge regression regularization '''
        gradient = (np.dot(X_train.T, loss) + ridge_lambda * theta) / len(X_train)
        theta = theta - alpha * gradient
        cost[i] = computeCost(X_train, Y_train, theta, ridge_lambda)
        
    return (theta, cost)

''' compute cost to be minimized with gradient descent '''
def computeCost(X, y, theta, ridge_lambda=0):

     ''' quadratic loss function '''
     ridge_factor = (ridge_lambda / (2*len(X))) * np.sum(np.square(theta)) 
     error = np.square((np.dot(X,theta.T) - y.T)) + ridge_factor
   
     '''2 is a normalization factor'''
     cost = np.sum(error) / 2*len(X)
     return cost




''' PREDICTIONS and METRICS '''
''' predict values of  test set and calculate risk estimate metrics'''
def predictRidge(X_valid, Y_valid, theta, plot=False):
    
    predictions=[]
   
    '''make prediction'''
    for i in range(len((Y_valid.values))):
        yhat = np.dot(X_valid.values[i], theta.T)        
        predictions.append(yhat)
        
        '''avoid div by 0'''
        if (Y_valid.values[i]==0):
            Y_valid.values[i] = Y_valid.values[i] + 1e-8
    
    predictions = pd.DataFrame(predictions)      
    min_max = datasetMinMax(predictions)
    rescaleDataset(predictions, min_max)       
    '''plot results'''
    rmse=rootMeanSquareError(Y_valid.values, predictions.values)
    maape = meanArctangentAbsolutePercentageError(Y_valid.values, predictions.values)
    
    if plot==True:        
        pred = pd.DataFrame(predictions)      
        min_max = datasetMinMax(pred)
        rescaleDataset(pred, min_max,min_max_target[0],min_max_target[1])
        Y_true = pd.DataFrame(Y_valid)      
        min_max = datasetMinMax(Y_true)
        rescaleDataset(Y_true, min_max,min_max_target[0],min_max_target[1]) 
        plt.figure(figsize=(12,5), dpi=100)
        plt.plot(Y_true.values, label='training')
        plt.plot(pred.values, label='actual')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()
        
    return rmse, maape, predictions

''' calculate scale dependent RMSE metric '''   
def rootMeanSquareError(actual, predictions):
    sum_error = 0.0
    for i in range(len(actual)):
        sum_error += np.square(predictions[i] - actual[i])
    root_mean_square_error = np.sqrt(sum_error / float(len(actual)))
    return root_mean_square_error

''' calculate MAAPE metric '''   
def meanArctangentAbsolutePercentageError(actual, predictions):
    maape = np.mean(np.arctan(np.abs((actual - predictions) / actual)))*100
    return maape

  

'''PCA - principal component analysis 
- calculate principal components
- define projected features keeping > 95% accuracy '''
def pca(data):
    
    data = data.drop(['total_bedrooms'], 1)
    data = data.drop(['households'], 1)
    cov_matrix = np.cov(data.values.T)
    ''' extract eigenvalues and eigenvectors'''
    values,vectors = np.linalg.eig(cov_matrix)
    explained_var = []

    for i in range(len(values)):
        explained_var.append(values[i]/np.sum(values))
    print('{0}\n{1}'.format(np.sum(explained_var),explained_var))
    
    plt.figure(figsize=(12,5), dpi=100)
    range(len(explained_var))
    plt.plot(np.arange(1,len(explained_var)+1),np.cumsum(explained_var), linestyle='--', marker='o')
    plt.xlabel('Principal components')
    plt.ylabel('Cumulative Explained variance')
    plt.show()
    print('autovettori')
    print(np.cumsum(explained_var))
    
    '''components to keep'''
    n = 5
    projections = []
    projections.append(data.values.dot(vectors.T[0]))
    new_features = pd.DataFrame(projections[0], columns=['PC1'])   
          
    ''' calculate new data values from components features '''                 
    for i in range(n-1):
        projections.append(data.values.dot(vectors.T[i+1]))
        new_features['PC{0}'.format(i+2)] = projections[i+1]

    return new_features, explained_var



''' Test start here ----------------------------------------------------------'''


''' adjust missing values for #bedrooms '''

estimated_bedrooms_values = imputation(house_price, missing_attr, related_attr)

''' redefine categorical feature ocean_proximity '''
#house_price_enc = oneHotEncoder(house_price, categoric_val, categoric_attr)
house_price_enc, avg = orderedIntegerEncoder(house_price, categoric_val, categoric_attr, target)


'''     rescale data           '''
house_price_scaled = house_price_enc.copy()
min_max = datasetMinMax(house_price_scaled)
min_max_target = min_max[8]
rescaleDataset(house_price_scaled, min_max)
#comment the next statement to avoid rescale data on Ridge Regression
house_price_enc = house_price_scaled


''' predict median house value '''

print('Predict median house value')
# features, target
Y = pd.DataFrame(house_price_enc[target])
X = house_price_enc.drop([target], 1)


''' Turn on/off comments to switch to nested '''
#print('Cross Validated Ridge Regression')
#pred = crossValidationRidge(X, Y)

print('Nested Cross Validated Ridge Regression')
pred = nestedCrossValidationRidge(X,Y)

print('Applying  PCA')
''' apply PCA '''
X_PCA, exp = pca(X)


''' apply Cross Validation Ridge on new dataset after PCA  '''

#print('Cross Validated Ridge Regression')
#pred = crossValidationRidge(X_PCA, Y)
print('Nested Cross Validated Ridge Regression')
pred = nestedCrossValidationRidge(X_PCA, Y)

