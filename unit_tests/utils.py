import os
import re


def get_pickle_by_id(id, MODELDIR):


    models = os.listdir(MODELDIR)
    modfile = filter(lambda x: int(re.search('_(-?\d*).pkl$', x)[1]) == int(id), models)
    try:
        modfile = list(modfile)[0]
    except:
        modfile = None
    return modfile