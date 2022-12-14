from flask import Flask, request, jsonify
import statsmodels.api as sma
import statsmodels as sm
import pandas as pd
import numpy as np
import json
from trainable_models import *


MODELDIR = './models'
MODLIST = [{'lib': 'sklearn', 'model': 'LinearRegression', 'name':'linreg'}, 
           {'lib': 'sklearn', 'model': 'DecisionTreeRegressor', 'name':'tree'}]

app = Flask(__name__)


@app.get("/listmod")
def list_model_classes():
    """
    Lists model classes that can be trained
    If a new model class is programmed to be trained add it to MODLIST variable
    """
    
    # read a JSON file with the same contents to not use "global"???
    global MODLIST
    return jsonify({'available models': MODLIST})


@app.post("/models")
def add_model():
    """
    Train a model using JSON data POSTed
    """
    id = None
    data = json.loads(request.data)
    model = data['model']
    
    # if model here is str then it is model specification
    if isinstance(model, str):
        model = for_train[model]
    # else it should be id to retrain the model, hyperparams and model class 
    # from JSON are overwritten if present (no need in allowing change of
    # hyperparameters as it will be the same as fitting a new model)
    else:
        try:
            id = model['id']
            data['hyperparams'], data['model'] = get_params_from_id(model['id'])
            model = data['model']
            model = for_train[model]
        except:
            return 'err 1', 500
    
    model(data['Y'], data['X'], data.get('hyperparams', None), id)
    return 'OK', 201


@app.post("/models/<model_id>")
def predict_delete(model_id):
    """
    POST with model deletion or getting model predict
    """
    id = int(model_id)
    print(request.data)
    data = json.loads(request.data)
    if data['action'] == 'delete':
        clear_ids(id)
        return 'Done', 201
    elif data['action'] == 'predict':
        models = os.listdir(MODELDIR)
        modfile = filter(lambda x: int(re.search('_(\d*).pkl$', x)[1]) == int(id), models)
        modfile = list(modfile)[0]
        with open(f'./models/{modfile}', 'rb') as handle:
            model = pickle.load(handle)
        print(list(model.predict(data['X'])))
        return jsonify({"prediction": list(model.predict(data['X']))}), 201
    else:
        return 'no/wrong action parameter', 500 