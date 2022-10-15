from flask import Flask, request, jsonify
import statsmodels.api as sma
import statsmodels as sm
import pandas as pd
import numpy as np


MODELDIR = './models'
MODLIST = ['sklearn:OLS', 'PyGAM:GAM']

app = Flask(__name__)

@app.get("/listmod")
def list_model_classes():
    global MODLIST
    return jsonify({'available models': MODLIST})


@app.get("/models")
def get_models():
    return "<p>Hello, World!</p>"
    
