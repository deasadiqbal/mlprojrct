import numpy as np
import sys
import pandas as pd
import os
from src.exception import ExceptionHandler
import dill
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_name= os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
       
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise ExceptionHandler(e,sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    model_report = {}
    try:
        for i, model_name in enumerate(models):
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            model_report[model_name] = r2

    except Exception as e:
        logging.error(e)
        raise ExceptionHandler(e)

    return model_report