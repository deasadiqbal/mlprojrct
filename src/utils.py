import numpy as np
import sys
import pandas as pd
import os
from src.exception import ExceptionHandler
import dill

def save_object(file_path, obj):
    try:
        dir_name= os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
       
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise ExceptionHandler(e,sys)