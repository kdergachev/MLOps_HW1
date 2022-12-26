from sklearn.linear_model import TweedieRegressor, LinearRegression
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from database_utils import *


MODELDIR = './models'

########################################
# ./models/ filenames are of type:     #
# YYYY-MM-DD_{modelname}_{modelid}.pkl #
########################################


def tree_fit(Y, X, hyperp, idx):
    """
    Fit a tree model with given hyperparameters/data
    
    INPUTS (determined by JSON format used)
    Y, X - endogenous and endogenous variables
    hyperp - hyperparams of the model
    id - id to resave model under
    RETURNS 
    0
    (saves the model in ./models/)
    """
    
    to_save = hyperp
    # take hyperparams for fit (others are in __init__)
    hyperp2 = {}
    hyperp2['sample_weight'] = hyperp.pop('sample_weight', None)
    hyperp2['check_input'] = hyperp.pop('check_input', None)
    
    # fit model
    if hyperp is None:
        model = DecisionTreeRegressor().fit(X, Y)
    else:
        model = DecisionTreeRegressor(**hyperp).fit(np.array(X), np.array(Y), **hyperp2)
    # create new attributes of model object so that hyperparams and name 
    # can be extracted
    model.ownvarofhyperparams = to_save
    model.ownvarofname = 'tree'
    
    # save the model
    db_add_model(model, 'tree', idx)
    return 0


def OLS_fit(Y, X, hyperp, idx):
    
    """
    Fit a linear regression model with given hyperparameters/data
    
    INPUTS (determined by JSON format used)
    Y, X - endogenous and endogenous variables
    hyperp - hyperparams of the model
    id - id to resave model under
    RETURNS 
    0
    (saves the model in ./models/)
    """
    
    # fit method hyperparams will be deprecated soon => not implemented
    # fit model
    if hyperp is None:
        model = LinearRegression().fit(X, Y)
    else:
        model = LinearRegression(**hyperp).fit(X, Y)
        
    # create new attributes of model object so that hyperparams and name 
    # can be extracted
    model.ownvarofhyperparams = hyperp
    model.ownvarofname = 'linreg'
    
    # save the model
    db_add_model(model, 'linreg', idx)
    return 0
    

for_train = {'linreg': OLS_fit, 'tree': tree_fit} # dict with trainable models


def get_params_from_id(idx):
    """
    Get model params and name from supplied model id
    
    INPUTS
    id: int - model id in the folder ./models/
    
    RETURNS
    model_hyperparameters: dict, model_name: str
    """
    model = db_fetch_model(idx)
    return model.ownvarofhyperparams, model.ownvarofname


def clear_ids(idx):
    """
    Remove all models with given id from ./models/
    
    INPUTS
    id: int - id to be removed
    
    RETURNS
    0
    (removes all files with given id from ./models/)
    
    """
    
    db_pure_delete(idx)
    return 0