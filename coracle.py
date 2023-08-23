# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 12:57:51 2021

@author: Sebastian Staab

Framework to analyse the microbiome of coral holobionts.
Includes the actual framework as well as a number of helper functions.
"""


import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from cfs import cfs
import asgl
import warnings
warnings.filterwarnings("ignore")





def weighting(x, test_x, weights):
    """
    converts x to a reduced dataset with only selected features
    
    Parameters
    ----------
    x : numpy array
        shape(n samples-1, k features), train features to train model
    test_x : numpy array
        shape(k features), test features to use for prediction
    y : numpy array
        shape(n samples-1), target variable
    weights : float
        coefficients of e.g. a penalized regression

    Returns
    -------
    x_new : numpy array
        converted x (only selected features)
    test_x_new : numpy array
        converted test x (only selected features)
    index : list
        indices of relevant features
    """
    #create empty new lists
    x_new=[]
    test_x_new=[]
    index=[]
    
    #iterate through the features and add only the ones with nonzero coefficients to the new lists
    for idx, val in enumerate(weights):
        if val != 0:
            x_new.append(x[:,idx])
            test_x_new.append(test_x[:,idx])
            index.append(idx)
    x_new = np.array(x_new).transpose()
    test_x_new = np.array(test_x_new).transpose()
    if x_new.size == 0:
        x_new = np.zeros((len(x), 1))
    if test_x_new.size == 0:
        x_new = np.zeros((len(x), 1)) 
    #if len(index) == 0:
        #print("all coefficients are zero'd out")
    return x_new, test_x_new, index



def alasso_rfr(X, test_X, y, lambdas):
    """
    Train alasso (feature selector) followed by a random forest regressor. 
    Create predicition based on test-x.

    Parameters
    ----------
    X : numpy array
        shape(n samples-1, k features), train features to train model
    test_X : numpy array
        shape(k features), test features to use for prediction
    y : numpy array
        shape(n samples-1), target variable
    lambdas : float
        value for the lambda (alpha) parameter of Lasso (penalization weight)

    Returns
    -------
    float
        prediction for lasso-rfr
    float
        prediction for lasso
    float
        prediction for alasso-rfr
    float
        prediction for alasso
    """
    #Lasso
    lasso = Lasso(alpha = lambdas)
    lasso.fit(X, y) #fit with training set
    predict_y_lasso = lasso.predict(test_X)[0]
    weight_lasso = lasso.coef_
    
    # get reduced datasets (only selected features)
    x_new_lasso, test_x_new_lasso, index_lasso = weighting(X, test_X, weight_lasso) 
    
    
    #Alasso
    reweight = (1/(np.abs(weight_lasso) + 0.0001))
    alasso = asgl.ASGL(model="lm", lambda1 = [lambdas], penalization="alasso", lasso_weights=reweight)
    alasso.fit(x=X, y=y.flatten())
    predict_y_alasso = alasso.predict(test_X)[0][0]
    weight_alasso = alasso.coef_[0][1:]
    
    # get reduced datasets (only selected features)
    x_new_alasso, test_x_new_alasso, index_alasso = weighting(X, test_X, weight_alasso) 
    
    
    
    # if no features are selected by both lasso and lasso return regression intercept as prediction
    if (len(index_lasso) == 0) and (len(index_alasso) == 0):
        intercept_lasso = lasso.intercept_[0]
        intercept_alasso = alasso.coef_[0][0]
        return intercept_lasso, intercept_lasso, intercept_alasso, intercept_alasso
    
    elif (len(index_lasso) == 0) and (len(index_alasso) > 0) : #if no features selected for lasso, return intercept
        intercept_lasso = lasso.intercept_[0]
        
        #Random Forest Regression alasso
        rfr = RandomForestRegressor()
        rfr.fit(x_new_alasso, y) #fit with training set
        predict_y_alasso_rfr = rfr.predict(test_x_new_alasso)[0]
        return intercept_lasso, intercept_lasso, predict_y_alasso_rfr, predict_y_alasso
    
    elif (len(index_lasso) > 0) and (len(index_alasso) == 0) : #if no features selected for alasso, return intercept
        #Random Forest Regression lasso
        rfr = RandomForestRegressor()
        rfr.fit(x_new_lasso, y) #fit with training set
        predict_y_lasso_rfr = rfr.predict(test_x_new_lasso)[0]
        
        intercept_alasso = alasso.coef_[0][0]
        return predict_y_lasso_rfr, predict_y_lasso, intercept_alasso, intercept_alasso
    
    else: # else train Random Forest Regression with reduced dataset for alasso and lasso
        #Random Forest Regression lasso
        rfr_lasso = RandomForestRegressor()
        rfr_lasso.fit(x_new_lasso, y) #fit with training set
        predict_y_lasso_rfr = rfr_lasso.predict(test_x_new_lasso)[0]
        
        #Random Forest Regression alasso
        rfr_alasso = RandomForestRegressor()
        rfr_alasso.fit(x_new_alasso, y) #fit with training set
        predict_y_alasso_rfr = rfr_alasso.predict(test_x_new_alasso)[0]
        
        # return predicition of lasso-rfr and of only lasso
        return predict_y_lasso_rfr, predict_y_lasso, predict_y_alasso_rfr, predict_y_alasso
    

def cfs_rfr(X, test_X, y):
    """
    Train cfs (feature selector) followed by a random forest regressor. 
    Create predicition based on test-x.

    Parameters
    ----------
    X : numpy array
        shape(n samples-1, k features), train features to train model
    test_X : numpy array
        shape(k features), test features to use for prediction
    y : numpy array
        shape(n samples-1), target variable
    lambdas : float
        value for the lambda (alpha) parameter of Lasso (penalization weight)

    Returns
    -------
    float
        prediction for cfs-rfr
    """
    
    #CFS - get reduced datasets (only selected features)
    selection =cfs(X, y)
    x_new = X[:, selection]
    test_x_new= test_X[:, selection]
     
    #Random Forest Regression
    rfr = RandomForestRegressor()
    rfr.fit(x_new, y) #fit with training set
    predict_y = rfr.predict(test_x_new)
    
    return predict_y



def rfr(X, test_X, y):
    """
    Train random forest regressor. 
    Create predicition based on test-x.

    Parameters
    ----------
    X : numpy array
        shape(n samples-1, k features), train features to train model
    test_X : numpy array
        shape(k features), test features to use for prediction
    y : numpy array
        shape(n samples-1), target variable
    lambdas : float
        value for the lambda (alpha) parameter of Lasso (penalization weight)

    Returns
    -------
    float
        prediction for rfr
    """

    # test dataset
    rfr = RandomForestRegressor()
    rfr.fit(X, y) #fit with training set
    predict_y = rfr.predict(test_X)
    
    return predict_y



def rfr_importance(x, y):
    """
    function to return the feature importance via a random forest regressor

    Parameters
    ----------
    x : numpy array
            shape(n samples, k features), independent variables
    y : numpy array
            shape(n samples), dependent variable

    Returns
    -------
    importance : numpy array
            feature importances
    """
    rfr = RandomForestRegressor()
    rfr.fit(x, y)
    importance = rfr.feature_importances_
    return importance



def alasso_rfr_importance(x, y, i):
    """
    function to return the feature importance via alasso (feature selector) and random forest regressor

    Parameters
    ----------
    x : numpy array
            shape(n samples, k features), independent variables
    y : numpy array
            shape(n samples), dependent variable
    i: float
            lambda parameter

    Returns
    -------
    importance : numpy array
            feature importances
    """
    #Lasso
    lasso = Lasso(alpha = i)
    lasso.fit(x, y) #fit with training set
    weight_lasso = lasso.coef_
    
    # get reduced datasets (only selected features)
    x_new_lasso, test_x_new_lasso, index_lasso = weighting(x, x, weight_lasso) 
    
    
    #Alasso
    reweight = (1/(np.abs(weight_lasso) + 0.0001))
    alasso = asgl.ASGL(model="lm", lambda1 = [i], penalization="alasso", lasso_weights=reweight)
    alasso.fit(x=x, y=y.flatten())
    weight_alasso = alasso.coef_[0][1:]
    
    # get reduced datasets (only selected features)
    x_new_alasso, test_x_new_alasso, index_alasso = weighting(x, x, weight_alasso)
    
    #Random Forest Regression for lasso    
    rfr_lasso = RandomForestRegressor()
    rfr_lasso.fit(x_new_lasso, y)
    
    #get importances
    importance_lasso = np.zeros(x.shape[1])
    j=0
    for idx, val in enumerate(weight_lasso):
        if val != 0:
            importance_lasso[idx] = rfr_lasso.feature_importances_[j]
            j+=1
            
        
    #Random Forest Regression for alasso    
    rfr_alasso = RandomForestRegressor()
    rfr_alasso.fit(x_new_alasso, y)
    
    #get importances
    importance_alasso = np.zeros(x.shape[1])
    j=0
    for idx, val in enumerate(weight_alasso):
        if val != 0:
            importance_alasso[idx] = rfr_alasso.feature_importances_[j]
            j+=1
    
    #get coefficients
    lasso_parameter = np.concatenate((lasso.intercept_, lasso.coef_))
    alasso_parameter = alasso.coef_[0]
    
    return importance_lasso, importance_alasso, lasso_parameter, alasso_parameter


def cfs_rfr_importance(x, y):
    """
    function to return the feature importance via a CFS (feature selector) and random forest regressor

    Parameters
    ----------
    x : numpy array
            shape(n samples, k features), independent variables
    y : numpy array
            shape(n samples), dependent variable

    Returns
    -------
    importance : numpy array
            feature importances
    """
    #CFS - get reduced datasets (only selected features)
    selec_cfs=cfs(x, y)
    x_new = x[:, selec_cfs]
                 
    #Random Forest Regression
    rfr_ = RandomForestRegressor()
    rfr_.fit(x_new, y) #fit with training set
    
    #get importances
    importance = np.zeros(x.shape[1])
    for idx, val in enumerate(selec_cfs):
        importance[val] = rfr_.feature_importances_[idx]

    return importance




###############################################################################
###
### Machine Learning Framework / Ensemble Feature Selection
### 
def coracle(x, y, alpha_l1 = 10**(-2.9), alpha_clr = 10**(-0.4)):
    """
    framework to analyse microbiomes. Optimized for the analysis of coral holobiont's 
    microbiome on the family- and order-level.

    Parameters
    ----------
    x : pandas dataframe,
            shape(n samples, k features), independent variables
    y : pandas dataframe, 
            shape(n samples), dependent variable
    alpha_l1 : float, optional
            lambda/alpha value for l1-lasso traverses. The default is 10**(-2.9) [pre-optimized].
    alpha_clr : float, optional
            lambda/alpha value for clr-lasso traverses. The default is 10**(-0.4) [pre-optimized].
    Alasso : boolean, optional
            Option to turn Adaptive Lasso off as it is a more exotic package. The default is True.

    Returns
    -------
    full_result : pandas dataframe
            full table including the final score, the performance matrix, feature importances and coefficients (+intercept)

    """
    ###########################################################################
    ### 1.. Check input
    # check for types
    if not isinstance(x, (pd.DataFrame)):
        raise TypeError("TypeError exception thrown. Expected pandas dataframe")
    if not isinstance(y, (pd.DataFrame)):
        raise TypeError("TypeError exception thrown. Expected pandas dataframe")
    if not isinstance(alpha_l1, (float)):
        raise TypeError("TypeError exception thrown. Expected float")
    if not isinstance(alpha_clr, (float)):
        raise TypeError("TypeError exception thrown. Expected float")

    
    # check for dimensions 
    if x.shape[0] != y.shape[0]:
        raise ValueError("ValueError exception thrown. Expected x and y to have the same number of samples")
    if x.ndim != 2 or y.ndim != 2:  
        raise ValueError("ValueError exception thrown. Expected x and y to have two dimensions")
    

    # delete empty columns
    x = x.loc[:, (x != 0).any(axis=0)]
    
    # change to numpy and save names
    microbiome = x.columns.values
    x = x.to_numpy()
    y = y.to_numpy()
    ###########################################################################
    
    
    ###########################################################################
    ### 2. Preprocessing
    ### Normalisation: relative abundance & centered log ratio
      
    #get relative abundance
    x_l1 = preprocessing.normalize(x, norm="l1")
    #centered log ratio
    min_value = np.amin(x)
    if min_value < 0: #in case of negative values they have to be adjusted since log can't handle negative values
        x_clr = np.log(x+1+abs(min_value))-np.log(x+1+abs(min_value)).mean(axis=1, keepdims=True)
        print("negative values have been detected, counts have been adjusted (absolute min value added to all counts) since log can't handle negative values. min value: ", min_value)
    else: 
        x_clr = np.log(x+1)-np.log(x+1).mean(axis=1, keepdims=True)
    
    
    #create some helpfull variables
    models = ["l1_rfr", "l1_lasso_rfr", "l1_alasso_rfr", "l1_cfs_rfr", "clr_rfr", "clr_lasso_rfr", "clr_alasso_rfr", "clr_cfs_rfr", "l1_lasso_coef", "l1_alasso_coef", "clr_lasso_coef", "clr_alasso_coef"]
    tra = len(models) #number of traverses
    n = len(y) #number of samples = number of cross validated splits
    ###########################################################################

    ###########################################################################
    ### 3. Leave-One-Out-Cross-Validation
    loo = LeaveOneOut()
    loo.get_n_splits(x)
    
    y_test_list = np.zeros(n) #store y_test
    y_predict_list = np.zeros((n, tra), dtype = float) #store predicted y
    
    j = 0
    for train_index, test_index in loo.split(x):
        
        #create train/validate sets
        X_train_l1, X_test_l1 = x_l1[train_index], x_l1[test_index]  #l1
        X_train_clr, X_test_clr = x_clr[train_index], x_clr[test_index] #clr
        y_train, y_test = y[train_index], y[test_index]
        
        #store y_test
        y_test_list[j] = y_test
        
        
        ### Models
        #Random Forest Regression and Lasso
        #l1
        y_predict_list[j, 0]  = rfr(X_train_l1, X_test_l1, y_train)
        y_predict_list[j, 1], y_predict_list[j, 8], y_predict_list[j, 2], y_predict_list[j, 9]  = alasso_rfr(X_train_l1, X_test_l1, y_train, alpha_l1)
        y_predict_list[j, 3]  = cfs_rfr(X_train_l1, X_test_l1, y_train)
        #clr
        y_predict_list[j, 4]  = rfr(X_train_clr, X_test_clr, y_train)
        y_predict_list[j, 5], y_predict_list[j, 10], y_predict_list[j, 6], y_predict_list[j, 11]  = alasso_rfr(X_train_clr, X_test_clr, y_train, alpha_clr)
        y_predict_list[j, 7]  = cfs_rfr(X_train_clr, X_test_clr, y_train)
        
        print(round((j+1)*100/(n+1)), "%")
        j+=1
    ###########################################################################
  
    ###########################################################################
    ### 4. get RFR importances and lasso/alasso coefficients
    feature_importance  =  np.zeros((x.shape[1], 8))
    lasso_parameter  =  np.zeros((x.shape[1]+1, 2))
    alasso_parameter  =  np.zeros((x.shape[1]+1, 2))
    
    #l1
    feature_importance[:, 0] = rfr_importance(x_l1, y)
    feature_importance[:, 1], feature_importance[:, 2], lasso_parameter[:,0], alasso_parameter[:,0] = alasso_rfr_importance(x_l1, y, alpha_l1)
    feature_importance[:, 3] = cfs_rfr_importance(x_l1, y)
    #clr
    feature_importance[:, 4] = rfr_importance(x_clr, y)
    feature_importance[:, 5], feature_importance[:, 6], lasso_parameter[:,1], alasso_parameter[:,1] = alasso_rfr_importance(x_clr, y, alpha_clr)
    feature_importance[:, 7] = cfs_rfr_importance(x_clr, y)
    ###########################################################################
    
    ###########################################################################
    ### 5. Scoring Function 
    
    #combine performance results
    r2 = np.zeros((tra))
    mse = np.zeros((tra))
    
    for ind in range(tra): #
        r2[ind] = r2_score(y_test_list, y_predict_list[:, ind])
        mse[ind] = mean_squared_error(y_test_list, y_predict_list[:, ind])    
    
    #the actual scoring function
    result = [r2, mse]
    result_score = result[0] #R²
    result_score[result_score < 0] = 0 #set R² to zero if negative (to not negatively affect the final score)
    score = (result_score[0:8] * feature_importance)
    score = np.sum(score/8, axis=1)
    
    #bring all results together
    microbiome_intercept = np.insert(microbiome, 0, "Intercept", axis=0)
    result = pd.DataFrame(result, index =["R2", "MSE"], columns=models)  
    score = pd.DataFrame(score, index = microbiome, columns=["score"])
    feature_importance = pd.DataFrame(feature_importance, index = microbiome, columns = models[0:8])
    
    #coefficients
    lasso_coef = pd.DataFrame(lasso_parameter, index = microbiome_intercept, columns=["l1_lasso_coef",  "clr_lasso_coef"])
    alasso_coef = pd.DataFrame(alasso_parameter, index = microbiome_intercept, columns=["l1_alasso_coef",  "clr_alasso_coef"])
    coef = pd.merge(lasso_coef, alasso_coef, left_index=True, right_index=True)
    
    #merge results to two final documents
    full = feature_importance.merge(coef, left_index=True, right_index=True, how = "right")
    full = score.merge(full, left_index=True, right_index=True, how = "right")
    intercept = full.loc["Intercept"].to_frame().transpose()
    full = pd.DataFrame(full.iloc[1:])
    full.sort_values(by=['score'], inplace=True, ascending=False) #sort
    rest = pd.concat([result, intercept]) #result.append(intercept) old
    full_result = pd.concat([rest, full])
    first_column = full_result.pop("score")
    full_result.insert(0, 'score', first_column)
    
    #additional weighting for the number of models that chose each feature
    weight = full_result[3:].drop(columns=["score", "l1_lasso_rfr", "clr_lasso_rfr"])
    weight[abs(weight) > 0] = 1
    full_result["score"] = full_result["score"] * weight.sum(axis=1)/8
    #sort again
    sortscore = full_result[3:].sort_values(by=['score'], ascending=False) #sort
    perf = full_result[:3]
    full_result = pd.concat([perf, sortscore])
    print(100, "%")
    #return the final score, all model results (performances & feature importances) and the lasso & alasso coefficients combined
    return full_result
    ###########################################################################




