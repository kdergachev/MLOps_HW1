from sklearn.linear_model import TweedieRegressor, LinearRegression
import pickle
import os
import re
from datetime import date
import numpy as np



def name_to_save(model):
    
    print(os.listdir('./models')[0])
    # re.search('_(\d+)\.pkl$', s).group(0)
    print(re.match('_(\d*).pkl$', os.listdir('./models')[0]))
    idx = [int(re.match('(\d*)$', i)[0]) for i in os.listdir('./models')]
    print(idx)
    if not idx:
        idx = 0
    else:
        idx = max(idx) + 1
    #idx = idx.max() + 1
    today = date.today()
    return f'{today.strftime("%Y-%m-%d")}_{model}_{idx}'


def GLM_fit(Y, X, hyperp):
    
    hyperp2 = {}
    hyperp2['sample_weight'] = hyperp.pop('sample_weight', None)
    model = TweedieRegressor(**hyperp).fit(X, Y, **hyperp2)
    
    name = name_to_save('GLM')
    with open(f'./models/{name}.pkl', 'wb') as handle:
        pickle.dump(model, handle)


def OLS_fit(Y, X, hyperp):
    
    hyperp2 = {}
    hyperp2['sample_weight'] = hyperp.pop('sample_weight', None)
    model = LinearRegression(**hyperp).fit(X, Y, **hyperp2)
    
    name = name_to_save('OLS')
    with open(f'./models/{name}.pkl', 'wb') as handle:
        pickle.dump(model, handle)
    

for_train = {'linreg': OLS_fit, 'GLM': GLM_fit}






    