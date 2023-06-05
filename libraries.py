from __future__ import absolute_import, division, print_function, unicode_literals

import sys, six, glob, math, os, logging, platform, random, requests, pytz, warnings
import psutil
from psutil import Process; 


#for measuring training time
import time
from datetime import date, datetime
from time import time 

import importlib
import pickle
import joblib


from IPython import display
from IPython.display import display, HTML

import seaborn as sns
from matplotlib import cm, gridspec, pyplot as plt

import numpy as np
import pandas as pd
# Visualising distribution of data
from pandas.plotting import scatter_matrix

from sklearn import (metrics, datasets, linear_model, preprocessing)
from sklearn.metrics import (
							accuracy_score, 
							classification_report, 
							plot_confusion_matrix, 
							recall_score, 
							f1_score, 
							roc_auc_score,
							cohen_kappa_score
							)
from sklearn.model_selection import ( 
							train_test_split, 
							cross_val_score, 
							KFold, 
							GridSearchCV, 
							StratifiedKFold 
							)
from sklearn.preprocessing import (
							LabelEncoder,
							LabelBinarizer, 
							OneHotEncoder, 
							MinMaxScaler,
							minmax_scale,
							MaxAbsScaler,
							StandardScaler,
							RobustScaler,
							Normalizer,
							QuantileTransformer
							)
from sklearn.pipeline import Pipeline
from sklearn.utils import class_weight
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier , ExtraTreesClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

try:
    from urllib.request import Request, urlopen  # Python 3
except ImportError:
    from urllib2 import Request, urlopen  # Python 2
    
import xgboost as xgb
from xgboost import XGBClassifier

from imblearn.ensemble import BalancedBaggingClassifier,BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline, make_pipeline

#from pystacknet.pystacknet import StackNetClassifier

import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow import feature_column
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adagrad, Adadelta, RMSprop, Adam

import keras
from keras import backend as K
from keras.models import Sequential, load_model 
from keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.wrappers.scikit_learn import KerasClassifier

#--
#------- System Functions 
def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)

def dnn_Hash_disGPU():
	## Seeding on GPU enable/disable for DNN Training
	#Pythgon hashed & disable GPU
	os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
	os.environ['CUDA_VISIBLE_DEVICES'] = ''
	os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'
	os.environ['PYTHONHASHSEED'] = '0'
	
	session_conf = tf.compat.v1.ConfigProto(
	   intra_op_parallelism_threads=1, 
	   inter_op_parallelism_threads=1
	)
	sess = tf.compat.v1.Session(
	   graph=tf.compat.v1.get_default_graph(), 
	   config=session_conf
	)


##-- SET environement Variables
sns.set(style="white", color_codes=True)

warnings.simplefilter(action='ignore', category=FutureWarning)

K.set_floatx('float64')

#Panda Options
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option("mode.chained_assignment", None)
pd.options.display.float_format = "{:,.2f}".format

set_tf_loglevel(logging.ERROR)
#tf.config.optimizer.set_jit(True)
#dnn_Hash_disGPU()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] ='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['PYTHONHASHSEED'] = '0'


sys.modules['sklearn.externals.six'] = six
sys.modules['sklearn.externals.joblib'] = joblib

# Read files from http
#- set requests header
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '\
           'AppleWebKit/537.36 (KHTML, like Gecko) '\
           'Chrome/75.0.3770.80 Safari/537.36'}
#------- General Functions

def nowtime():

  tz = pytz.timezone('Europe/Paris')
  now = datetime.now(tz)
  current_time = now.strftime("%Y-%m-%d %H:%M:%S")
  current_display_time = now.strftime("%H:%M")
  current_timestamp = datetime.timestamp(now)

  return current_time, current_display_time, current_timestamp

def getDiffMinutes(current_time,matchTime):
    
    fmt = '%Y-%m-%d %H:%M:%S'
    d1 = pd.to_datetime(matchTime, format=fmt)
    d2 = datetime.strptime(current_time, fmt)

    # calculate unix datetime
    d1_ts = ((matchTime - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).astype(float)
    d2_ts = int(time.mktime(d2.timetuple()))
    
    #-
    diffminutes=(int(d2_ts)-d1_ts) / 60
    
    return diffminutes

def reset_random_seeds():
    # Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(42)
    #Set `python` built-in pseudo-random generator at a fixed value
    random.seed(42)
    tf.random.set_seed(42)



