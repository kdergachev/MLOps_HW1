from sklearn.linear_model import TweedieRegressor, LinearRegression
import pickle
import os
import re
from datetime import date
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import numpy as np


MODELDIR = './models'

########################################
# ./models/ filenames are of type:     #
# YYYY-MM-DD_{modelname}_{modelid}.pkl #
########################################


def name_to_save(model, idx):
    
    """
    Get a name to save the model under.
    
    INPUTS
    model: str - string identifying model
    idx (optional): int - id of the model if it is refit from ones fitted before
    if None the id is chosen automatically
    
    RETURNS
    str - 'YYYY-MM-DD_{modelname}_{modelid}' where model id is the smallest 
    possible id or the one supplied in input
    """
    
    # get id if not supplied with one
    if idx is None:
        idx = [int(re.search('_(\d*).pkl$', i)[1]) for i in os.listdir(MODELDIR)]
        if not idx:
            idx = 0
        else:
            idx = max(idx) + 1
    # remove model with given id (old one)
    else:
        clear_ids(idx) # now hope no errors come up further
    today = date.today()
    return f'{today.strftime("%Y-%m-%d")}_{model}_{idx}'


def tree_fit(Y, X, hyperp, id):
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
    name = name_to_save('tree', id)
    with open(f'./models/{name}.pkl', 'wb') as handle:
        pickle.dump(model, handle)
    return 0


def OLS_fit(Y, X, hyperp, id):
    
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
    name = name_to_save('OLS', id)
    with open(f'./models/{name}.pkl', 'wb') as handle:
        pickle.dump(model, handle)
    return 0
    

for_train = {'linreg': OLS_fit, 'tree': tree_fit} # dict with trainable models


def get_params_from_id(id):
    """
    Get model params and name from supplied model id
    
    INPUTS
    id: int - model id in the folder ./models/
    
    RETURNS
    model_hyperparameters: dict, model_name: str
    """
    models = os.listdir(MODELDIR)
    modfile = filter(lambda x: int(re.search('_(\d*).pkl$', x)[1]) == int(id), models)
    modfile = list(modfile)[0] # to remove the first (hopefully the only) match
    with open(f'./models/{modfile}', 'rb') as handle:
        model = pickle.load(handle)
    return model.ownvarofhyperparams, model.ownvarofname


def clear_ids(id):
    """
    Remove all models with given id from ./models/
    
    INPUTS
    id: int - id to be removed
    
    RETURNS
    0
    (removes all files with given id from ./models/)
    
    """
    
    models = os.listdir(MODELDIR)
    modfiles = filter(lambda x: int(re.search('_(\d*).pkl$', x)[1]) == int(id), models)
    [os.remove(f'{MODELDIR}/{i}') for i in modfiles]
    return 0