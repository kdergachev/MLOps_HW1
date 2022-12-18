import os
import sys
import inspect

# insert parent folder to folders from which python can import 
# I found no better way to do this
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from trainable_models import *
import json
from sklearn.linear_model import TweedieRegressor, LinearRegression
import time
from utils import *


os.chdir('../')
MODELDIR = './models'

# я уже не выдерживаю. Тест каждой функции требует теста предыдущей чтобы быть
# честным/не оставлять после себя ничего, 
# а как правильно тестить trainable_models::name_to_save?????

# Предположим что эта функция верна по определению (там нечего тестить)


def test_clear_ids():
    """
    Test if clear_ids actually deletes files with given id
    Necessary for further tests so that tests leave no trace in the repo
    """
    
    global MODELDIR
    
    # создать элемент (id=-1 чтобы не пересечься с нетестовыми моделями)
    id = -1
    name = name_to_save('linreg', -1)
    before = os.listdir(MODELDIR)
    
    # записать в него что-то
    with open(f'{MODELDIR}/{name}.pkl', 'w') as f:
        f.write('Pretend this is binary')
    
    assert os.listdir(MODELDIR)[-1] == f'{name}.pkl'
    
    time.sleep(2) # посмотреть глазами что файл создаётся
    # удалить
    clear_ids(-1)
    assert os.listdir(MODELDIR) == before
    

def test_OLS_fit():
    """
    Tests OLS_fit function.
    no hyperparameters here (there are few for sklearn OLS anyway)
    tree will use hyperparameters though
    could be extended for some more examples/use of hyperparams
    """

    with open('./almost_system_tests/test1.json', 'r') as f:
        data = json.loads(f.read())

    OLS_fit(data['Y'], data['X'], data.get('hyperparams', None), -1)
    
    name = get_pickle_by_id(-1, MODELDIR)
    # проверил что она сохранена (ассёртом а не ошибкой)
    assert name is not None
    
    with open(f'./models/{name}', 'rb') as handle:
        model = pickle.load(handle)
    
    time.sleep(3)
    # просто данные из блокнота (в almost_system_tests)
    testdata = np.array([30.00384338, 25.02556238, 30.56759672, 28.60703649, 27.94352423,
                         25.25628446, 23.00180827, 19.53598843, 11.52363685, 18.92026211,
                         18.99949651, 21.58679568, 20.90652153, 19.55290281, 19.28348205,
                         19.29748321, 20.52750979, 16.91140135, 16.17801106, 18.40613603])
    pred = model.predict(data['X'][:20])
    # проверил результат (предпологая что isclose результат даст одна и та же 
    # модель. У них там есть __eq__, но я не докопался как он работает
    # вроде ни в самом классе ни в том из чего он наследуется нет её, но у 
    # объекта есть) 
    assert np.isclose(testdata, pred).all()
    
    clear_ids(-1)
    
    name = get_pickle_by_id(-1, MODELDIR)
    # проверить что удалил (не совсем часть теста функции, но надо проверить)
    assert name is None
    

def test_tree_fit():
    """
    Test tree_fit function
    same as before but with hyperp and tree rather than OLS
    """

    with open('./almost_system_tests/test2.json', 'r') as f:
        data = json.loads(f.read())

    tree_fit(data['Y'], data['X'], data.get('hyperparams', None), -1)
    
    name = get_pickle_by_id(-1, MODELDIR)
    
    assert name is not None
    
    with open(f'./models/{name}', 'rb') as handle:
        model = pickle.load(handle)
    
    time.sleep(3)

    # просто данные из блокнота (в almost_system_tests)
    testdata = np.array([23.46666667, 20.76598639, 34.544     , 34.544     , 34.544     ,
                         28.7       , 20.76598639, 22.1       , 16.9125    , 16.9125    ,
                         16.9125    , 20.76598639, 22.96666667, 20.76598639, 20.76598639,
                         20.76598639, 23.85714286, 19.40833333, 20.76598639, 20.76598639])

    pred = model.predict(data['X'][:20])
    print(pred)
    assert np.isclose(testdata, pred).all()
    
    clear_ids(-1)
    
    name = get_pickle_by_id(-1, MODELDIR)
    assert name is None
    






