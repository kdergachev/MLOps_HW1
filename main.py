from flask import Flask, request, jsonify
import statsmodels.api as sma
import statsmodels as sm
import pandas as pd
import numpy as np
import json
from trainable_models import *


MODELDIR = './models'
MODLIST = ['sklearn:OLS', 'PyGAM:GAM']

app = Flask(__name__)

@app.get("/listmod")
def list_model_classes():
    global MODLIST
    return jsonify({'available models': MODLIST})


@app.post("/models")
def add_model():
    data = json.loads(request.data)
    model = data['model']
    model = for_train[model]
    model(data['Y'], data['X'], data['hyperparams'])
    return 'OK', 201


@app.post("/models")
def add_model():
    data = json.loads(request.data)