from sklearn.linear_model import TweedieRegressor, LinearRegression
import pickle
import os
import re
from datetime import date
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import numpy as np


MODELDIR = './models'

def name_to_save(model, idx):
    
    # re.search('_(\d+)\.pkl$', s).group(0)
    if idx is None:
        idx = [int(re.search('_(\d*).pkl$', i)[1]) for i in os.listdir(MODELDIR)]
        if not idx:
            idx = 0
        else:
            idx = max(idx) + 1
    else:
        clear_ids(idx) # now hope no errors come up further
    today = date.today()
    return f'{today.strftime("%Y-%m-%d")}_{model}_{idx}'


def tree_fit(Y, X, hyperp, id):
    
    to_save = hyperp
    hyperp2 = {}
    hyperp2['sample_weight'] = hyperp.pop('sample_weight', None)
    hyperp2['check_input'] = hyperp.pop('check_input', None)
    
    if hyperp is None:
        model = DecisionTreeRegressor().fit(X, Y)
    else:
        model = DecisionTreeRegressor(**hyperp).fit(np.array(X), np.array(Y), **hyperp2)
    
    model.ownvarofhyperparams = to_save
    model.ownvarofname = 'tree'
    
    name = name_to_save('tree', id)
    with open(f'./models/{name}.pkl', 'wb') as handle:
        pickle.dump(model, handle)


def OLS_fit(Y, X, hyperp, id):
    
    # will become deprecated soon
    #hyperp2 = {}
    #hyperp2['sample_weight'] = hyperp.pop('sample_weight', None)
    #model = LinearRegression(**hyperp).fit(X, Y, **hyperp2)
    
    if hyperp is None:
        model = LinearRegression().fit(X, Y)
    else:
        model = LinearRegression(**hyperp).fit(X, Y)
    
    model.ownvarofhyperparams = hyperp
    model.ownvarofname = 'linreg'
    
    name = name_to_save('OLS', id)
    with open(f'./models/{name}.pkl', 'wb') as handle:
        pickle.dump(model, handle)
    

for_train = {'linreg': OLS_fit, 'tree': tree_fit}


def get_params_from_id(id):
    
    models = os.listdir(MODELDIR)
    modfile = filter(lambda x: int(re.search('_(\d*).pkl$', x)[1]) == int(id), models)
    modfile = list(modfile)[0]
    with open(f'./models/{modfile}', 'rb') as handle:
        model = pickle.load(handle)
    return model.ownvarofhyperparams, model.ownvarofname


def clear_ids(id):

    models = os.listdir(MODELDIR)
    modfiles = filter(lambda x: int(re.search('_(\d*).pkl$', x)[1]) == int(id), models)
    [os.remove(f'{MODELDIR}/{i}') for i in modfiles]
    return 0
    
    