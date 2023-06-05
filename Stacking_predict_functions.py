

import importlib
from libraries import *

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import PredictPlayerRating

importlib.reload(PredictPlayerRating)
from PredictPlayerRating import * 

import pytz
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def reset_random_seeds():
  #Set `tensorflow` pseudo-random generator at a fixed value    
  tf.random.set_seed(1234)
  # Set `numpy` pseudo-random generator at a fixed value
  np.random.seed(1234)
  #Set `python` built-in pseudo-random generator at a fixed value
  random.seed(1234)

def nowtime():

  tz = pytz.timezone('Europe/Paris')
  now = datetime.now(tz)
  current_time = now.strftime("%Y-%m-%d %H:%M:%S")
  current_display_time = now.strftime("%H:%M")
  current_timestamp = datetime.timestamp(now)

  return current_time, current_display_time, current_timestamp

def getDataToPredict(offset,date):
    
    print("#### Impoprt match data")
    path = "https://web-concepts.fr/soccer/win-pronos/pronos-R-data-dev.php?action=predict_dataset&offset="+str(offset)+"&startDate="+str(date)
    
    df = pd.read_csv(path, sep=",", encoding = "ISO-8859-1")
    #print("imported dataframe shape=",df.shape)
    #print(path)
    try:    
        df = df.sort_values(by=['event_timestamp']).reset_index(drop=True)
    except:
        r=requests.get(path)
        print(path)
        display(HTML(r.text))
        
    current_time, current_display_time, current_timestamp = nowtime()
    
    print("  -> import dataset : ok", len(df), " matches at ",current_time )
    #print("  -> import dataset : ok", len(df), " matches at ",current_time , " with shape: ",df.shape)
    return df

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

#Create an input pipeline using tf.data
# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, label_name, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop(label_name)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def setFeatureColumns(dataframe):
    # numeric cols
    feature_columns = []
    feature_numeric=[]
    feature_numeric=[
        'round', 
        'HTGS', 'HTGC', 'HTP', "HTGD", 
        'ATGS', 'ATGC', 'ATP', "ATGD",
        
        'homeWin_halftime','awayWin_halftime',
        
        'pred_match_winner_1','pred_match_winner_2','pred_match_winner_N','pred_match_winner_1N','pred_match_winner_N2',
        
        'HTF', 'ATF', 
        'pred_under_over', 'pred_goals_home', 'pred_goals_away',
        
        'HTPlayerDScore','HTPlayerMScore','HTPlayerFScore','ATPlayerDScore','ATPlayerMScore','ATPlayerFScore',
        
        'BW1', 'BWN', 'BW2',
        'BOver1_5goals', 'BUnder1_5goals','BOver2_5goals', 'BUnder2_5goals','BOver3_5goals', 'BUnder3_5goals',
        'BLDEM_YES', 'BLDEM_NO',
        
        'W1', 'WN', 'W2', "DiffWin12", "DiffWin1N", "DiffWin2N",
        'Over0_5goals', 'Over1_5goals', 'Over2_5goals', 'Over3_5goals', 'Over4_5goals', 
        'LDEM_YES',
        
        'HTPlayerGScore',
        'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
        'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
        'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
        'ATPlayerGScore',
        'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
        'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
        'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score'
    ]
    for keys in feature_numeric:
        feature_columns.append(feature_column.numeric_column(keys))

    features_categorical=[]

    # categorical indicator cols
    features_categorical=[
        'country_name', 
        'league_name', 
        'homeTeam', 
        'awayTeam',
        
        'HTPlayerG',
        'HTPlayerD1','HTPlayerD2','HTPlayerD3','HTPlayerD4','HTPlayerD5','HTPlayerD6',
        'HTPlayerM1','HTPlayerM2','HTPlayerM3','HTPlayerM4','HTPlayerM5','HTPlayerM6','HTPlayerM7',
        'HTPlayerF1','HTPlayerF2','HTPlayerF3','HTPlayerF4',
        
        'ATPlayerG',
        'ATPlayerD1','ATPlayerD2','ATPlayerD3','ATPlayerD4','ATPlayerD5','ATPlayerD6',
        'ATPlayerM1','ATPlayerM2','ATPlayerM3','ATPlayerM4','ATPlayerM5','ATPlayerM6','ATPlayerM7',
        'ATPlayerF1','ATPlayerF2','ATPlayerF3','ATPlayerF4'
    ]
    for keys in features_categorical:
        list=dataframe[keys].unique().tolist()
        tmp = feature_column.categorical_column_with_vocabulary_list(
            key=keys,
            vocabulary_list=list)
        thal_one_hot = feature_column.indicator_column(tmp)
        feature_columns.append(thal_one_hot)
    return feature_columns

def CreateModel(feature_columns,  n_class, optimizer,  selected_model,   dropout,  regul  ):
    

    ## CREATE THE DNN
    #Create a feature layer : input the feature_columns to our Keras model
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    activation_func='relu'
    
    #Create the DNN Sequential layers
    model=SetModel(n_class,activation_func,feature_layer,model=selected_model,dropout=dropout,regul=regul)


    ## Define loss function = 'categorical_crossentropy' or 'mean_squared_error'
    #Loss functions
    if n_class>1:
        loss_function = tf.keras.losses.sparse_categorical_crossentropy
    else:
        loss_function="binary_crossentropy"
        

    model.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=['accuracy'])

    #----
    return model

def TrainModel(
    feature_columns, 
    label_name, 
    n_class, 
    train_df, 
    valid_df, 
    learning_rate, 
    optimizer, 
    epochs,
    batch_size,
    verbose,
    decay,
    selected_model,
    dropout,
    regul,
    momentum
    ):
    reset_random_seeds()
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y-%m-%d")
    
    #----
    ## Create the DNN model
    model = CreateModel(feature_columns,n_class,optimizer = optimizer,selected_model=selected_model,dropout=dropout,regul=regul)
    
    #----
    #convert Dataframe to datasets
    train_ds = df_to_dataset(train_df, label_name, batch_size=batch_size)
    valid_ds = df_to_dataset(valid_df, label_name, batch_size=batch_size, shuffle=False)
    
    #----
    
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "trainingModels/lab/model-cp-"+label_name+"-v3.ckpt"
    checkpoint_dir  = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 20 epochs
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=verbose, 
        save_weights_only=True,
        period=5)
    
    #-- Time decay
    if decay=="step":
        loss_history = LossHistory()
        #Callback Step decay learning rate
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [loss_history, lrate,cp_callback]
    elif decay=="exponential":
        loss_history = exp_decay_LossHistory()
        #Callback Step decay learning rate
        lrate = LearningRateScheduler(exp_decay)
        callbacks_list = [loss_history, lrate, cp_callback]
    else :
        callbacks_list = [cp_callback]
    #----
    print("\n\n    label   : ",label_name)
    print("    optimizer   : ",optimizer)
    print("    model       : ",selected_model)
    if("L1" in selected_model)    : print("    Ruglarization : L1")
    if("L2" in selected_model)    : print("    Ruglarization    : L2")
    if("Regul" in selected_model) : print("    Ruglarization rate: ",regul)
    print("    batch_size  : ",batch_size)
    print("    dropout     : ",dropout)
    print("    momentum    : ",momentum,"\n")
    #----
    history = model.fit( 
        train_ds,
        validation_data=valid_ds,
        verbose=verbose,
        epochs=epochs,
        callbacks=callbacks_list)
    
    #----
    print("\n\n    Training validation:  ")
    print("\n  ==> Validation Accuracy: ", history.history["val_accuracy"][-1], 
          " || Validation Loss: ", history.history["val_loss"][-1])
    
    # Save the model
    #model.save("trainingModels/model-"+label_name+".h5")
    #model.save_weights(checkpoint_path.format(epoch=0))


    #----
    return model,history

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
       self.losses = []
       self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(len(self.losses)))
        print('step decay lr:', exp_decay(len(self.losses)))

# define step decay function
class exp_decay_LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(len(self.losses)))
        print('exp decay lr:', exp_decay(len(self.losses)))

def step_decay(epoch):
   initial_lrate = 0.1
   drop = 0.3
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

def exp_decay(epoch):
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate

def resample(types,df):
    if types=="under":
        # Class count
        count_class_0, count_class_1, count_class_2 = df["FTR"].value_counts()

        # Divide by class
        df_class_0 = df[df["FTR"] == 0]
        df_class_1 = df[df["FTR"] == 1]
        df_class_2 = df[df["FTR"] == 2]
        
        df_class_1_under = df_class_1.sample(count_class_2)
        dfout = pd.concat([df_class_0, df_class_1_under, df_class_2], axis=0)

        print('Random under-sampling:')

        
    if types=="over":
        smote = SMOTE(random_state=2)
        x_train = df.drop([label_name],1)
        y_train = df[label_name]
        X_sm, y_sm = smote.fit_sample(x_train, y_train)
        dfout = pd.DataFrame(X_sm)
        dfout[label_name]=y_sm
        print("smote over-sampling::")
        
    print(dfout[label_name].value_counts())
        
    dfout["FTR"].value_counts().plot(kind='bar', title='Count (target)');
    
    return dfout

def SetModel(
    n_class,
    activation_func,
    feature_layer,
    model,
    dropout=0.09,
    regul=0.0001,
    momentum=0.99):
    
    #----
    if  n_class>1:
        activationFn="softmax"
    else:
        activationFn="sigmoid"
    #----
    
    #----
    model_small = tf.keras.Sequential([
      feature_layer,
      layers.Dense(64, activation=activation_func),
      layers.Dense(32, activation=activation_func),
      layers.Dense(16, activation=activation_func),
      layers.Dense(8, activation=activation_func),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_medium = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func),
      layers.Dense(128, activation=activation_func),
      layers.Dense(64, activation=activation_func),
      layers.Dense(32, activation=activation_func),
      layers.Dense(16, activation=activation_func),
      layers.Dense(8, activation=activation_func),
      layers.Dense(n_class, activation=activationFn)
    ])
    #----
    model_large = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func),
      layers.Dense(256, activation=activation_func),
      layers.Dense(256, activation=activation_func),
      layers.Dense(256, activation=activation_func),
      layers.Dense(256, activation=activation_func),
      layers.Dense(n_class, activation=activationFn)
    ])
    #----
    model_vlarge = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(n_class, activation=activationFn)
    ])
    #---- Regularization models
    #---- BatchNormalization
    
    model_medium_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(),
      layers.Dense(128, activation=activation_func),
        layers.BatchNormalization(),
      layers.Dense(64, activation=activation_func),
        layers.BatchNormalization(),
      layers.Dense(32, activation=activation_func),
        layers.BatchNormalization(),
      layers.Dense(16, activation=activation_func),
        layers.BatchNormalization(),
      layers.Dense(8, activation=activation_func),
        layers.BatchNormalization(),
      layers.Dense(n_class, activation=activationFn)
    ])
    
    model_large_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(n_class, activation=activationFn)
    ])
    
    model_large_layer1_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(momentum=momentum),
      layers.Dense(256, activation=activation_func),
      layers.Dense(256, activation=activation_func),
      layers.Dense(256, activation=activation_func),
      layers.Dense(256, activation=activation_func),
      layers.Dense(256, activation=activation_func),
      layers.Dense(n_class, activation=activationFn)
    ])
    #---- Dropout
    model_small_drop = tf.keras.Sequential([
      feature_layer,
      layers.Dense(64, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(32, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(16, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(8, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_medium_drop = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(128, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(64, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(32, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(16, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_mlarge_drop = tf.keras.Sequential([
      feature_layer,
      layers.Dense(200, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(200, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(200, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(200, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(200, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_large_drop = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    #---- L2Regul
    model_vsmall_L2Regul = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(128, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(32, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_small_L2Regul = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_large_L2Regul = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_vlarge_L2Regul = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    
    
    #---- BatchNormalization
    model_large_L2Regul_drop_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_large_L2Regul_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_vlarge_L2Regul_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(n_class, activation=activationFn)
    ])
    
    model_large_L1L2Regul_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1_l2(regul)),
      layers.Dense(n_class, activation=activationFn)
    ])
    
    model_large_Droupout_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(),
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    
        
        
    #---- L1Regul
    model_medium_L1Regul = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])

    model_large_L1Regul = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func, kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    #---
    if model=="model_small":return model_small
    if model=="model_medium":return model_medium
    if model=="model_large":return model_large
    if model=="model_vlarge":return model_vlarge
    
    if model=="model_small_drop":return model_small_drop
    if model=="model_medium_drop":return model_medium_drop
    if model=="model_mlarge_drop":return model_mlarge_drop
    if model=="model_large_drop":return model_large_drop
    
    if model=="model_vsmall_L2Regul":return model_vsmall_L2Regul
    if model=="model_small_L2Regul":return model_small_L2Regul
    if model=="model_large_L2Regul":return model_large_L2Regul
    if model=="model_vlarge_L2Regul":return model_vlarge_L2Regul
    
    if model=="model_medium_L1Regul":return model_medium_L1Regul
    if model=="model_large_L1Regul":return model_large_L1Regul
    
    if model=="model_medium_BatchNormalization":return model_medium_BatchNormalization
    if model=="model_large_BatchNormalization":return model_large_BatchNormalization
    if model=="model_large_layer1_BatchNormalization":return model_large_layer1_BatchNormalization
    if model=="model_large_Droupout_BatchNormalization": return model_large_Droupout_BatchNormalization
    
    if model=="model_large_L2Regul_BatchNormalization":return model_large_L2Regul_BatchNormalization
    if model=="model_large_L2Regul_drop_BatchNormalization" : return model_large_L2Regul_drop_BatchNormalization
    if model=="model_vlarge_L2Regul_BatchNormalization":return model_vlarge_L2Regul_BatchNormalization
    
    if model=="model_large_L1L2Regul_BatchNormalization": return model_large_L1L2Regul_BatchNormalization

 
def ftrRule(x):
        if x['goalDiff'] >0:
            return 1
        elif  x['goalDiff'] <0:
            return 2
        elif  x['goalDiff'] ==0:
            return 0
        else:
            return -1
        
def ratingRanges(x,col):
  if (x[col] >= 3 and x[col] < 3.25)  :
      return 3
  elif (x[col] > 3.25 and x[col] < 3.75)  :
      return 3.5
  elif (x[col] > 3.75 and x[col] < 4.25)  :
      return 4.0
  elif (x[col] > 4.25 and x[col] < 4.75)  :
      return 4.5
  elif (x[col] > 4.75 and x[col] < 5.25 ) :
      return 5.0
  elif (x[col] > 5.25 and x[col] < 5.75)  :
      return 5.5
  elif (x[col] > 5.75 and x[col] < 6.25 ) :
      return 6.0
  elif (x[col] > 6.25 and x[col] < 6.75)  :
      return 6.5
  elif (x[col] > 6.75 and x[col] < 7.25)  :
      return 7.0
  elif (x[col] > 7.25 and x[col] < 7.75)  :
      return 7.5
  elif (x[col] > 7.75 and x[col] < 8.25)  :
      return 8.0
  elif (x[col] > 8.25 and x[col] < 8.75)  :
      return 8.5
  elif (x[col] > 8.75 and x[col] < 9.25)  :
      return 9.0
  elif (x[col] > 9.25 and x[col] < 9.75)  :
      return 9.5
  elif (x[col] > 9.75) :
      return 10.0
  else:
      return 0

def ratingRanges_v3(x,col):
  if (x[col] > 2.5 and x[col] <= 3.5)  :
      return 3
  elif (x[col] > 3.5 and x[col] <= 4.5)  :
      return 4
  elif (x[col] > 4.5 and x[col] <= 5.5)  :
      return 5
  elif (x[col] > 5.5 and x[col] <= 6.5)  :
      return 6
  elif (x[col] > 6.5 and x[col] <= 7.5)  :
      return 7
  elif (x[col] > 7.5 and x[col] <= 8.5)  :
      return 8
  elif (x[col] > 8.5 and x[col] <= 9.5)  :
      return 9
  elif (x[col] > 9.5)   :
      return 10
  else:
      return 0   

def splitData(dfTemp,label_name,splitratio,prints=True,TrainModel=True,encode=True):
    
    lab_enc = LabelEncoder()
    
    ### Set the label : NextTrend
    TrendX_all = dfTemp.drop([label_name],1)

    TrendY_all = dfTemp[label_name]
    if(encode==True):
        TrendY_all = lab_enc.fit_transform(TrendY_all)

    X_train = []
    X_test  = []
    y_train = []
    y_test  = []
    
    #### Split data to Training/Test
    
    try:
        if(TrainModel==True):
            
            # Shuffle and split the dataset into training and testing set.
            X_train, X_test, y_train, y_test = train_test_split(TrendX_all, TrendY_all,  test_size = int(len(dfTemp)*splitratio),  random_state = 1234, stratify = TrendY_all)
    except:
        if(TrainModel==True):
            
            # Shuffle and split the dataset into training and testing set.
            print("Warning: ",label_name," Not Stratified")
            X_train, X_test, y_train, y_test = train_test_split(TrendX_all, TrendY_all,  test_size = int(len(dfTemp)*splitratio), random_state = 1234)
        
    #----
    if prints==True:
        print(len(X_train), 'train examples')
        print(len(X_test) , 'validation examples')

    return TrendX_all,TrendY_all,X_train, X_test, y_train, y_test,lab_enc

def formatData(dataframeInput,label_name,split_ratio):
    #################### PREPARE THE DATA  #####################
    #Train dataframe
    print("############### Training ",label_name," ###############")
    dataframe=predict_prepareDataForNN(dataframeInput,label_name)
    print("Preparing Data... ",dataframe.shape)
    display(dataframe.head())
    
    #if(label_name=="FTR"): 
    #    dataframe=resample("under",dataframe)
        
    #print(dataframe[label_name].value_counts())
    
    #Set the feature columns
    feature_columns = setFeatureColumns(dataframe)
    print("end building feature columns")

    #Split the dataframe into train, validation
    train_max_row = int(dataframe.shape[0]*split_ratio)

    #----
    train_df=dataframe.iloc[:train_max_row]
    valid_df = dataframe.iloc[train_max_row:]

    #----
    print(len(train_df), 'train examples')
    print(len(valid_df), 'validation examples')    
    return feature_columns,train_df,valid_df

def predict_prepareDataForNN(dataframeInput,label_name,HT=False):
    
    dataframe=dataframeInput.copy()
    #Remove unnecessary columns
    dataframe=dataframe.drop([
        "homeUpdated",
        "awayUpdated",
        "statusShort",
        "status"
    ], axis=1)
    
    
    #Format conversions
    dataframe["event_date"]   = pd.to_datetime(dataframe["event_date"], format='%Y-%m-%d')
    dataframe["LDEM_YES"]     = pd.to_numeric(dataframe["LDEM_YES"], errors='coerce')
    dataframe["Over2_5goals"] = pd.to_numeric(dataframe["Over2_5goals"], errors='coerce')
    
    dataframe["round"]        = dataframe["round"].fillna(0.0).astype(int)
    dataframe["HTF"]          = dataframe["HTF"].fillna(0.0).astype(int)
    dataframe["ATF"]          = dataframe["ATF"].fillna(0.0).astype(int)
    
    #Add New Data to the dataset
    dataframe["totgoal"]  = dataframe['goalsHomeTeam']+dataframe['goalsAwayTeam']
    dataframe["goalDiff"] = dataframe['goalsHomeTeam']-dataframe['goalsAwayTeam']
    dataframe["FTHG"]     = dataframe['goalsHomeTeam']
    dataframe["FTAG"]     = dataframe['goalsAwayTeam']
    #dataframe["Season"]=dataframe['season']
    
    #CREATE BINARY OUTPUTS
    
    #FixtureTimeResult values =>  retrun 1 if team1; 2 if team2 ; 0 if draw ; else -1
    dataframe['FTR'] = dataframe.apply(ftrRule, axis=1)
    
    if label_name =="totgoal0":    
        dataframe["totgoal0"] = np.where( dataframe["totgoal"]==0, 1, 0)
    if label_name =="totgoal1":    
        dataframe["totgoal1"] = np.where( dataframe["totgoal"]==1, 1, 0)
    if label_name =="totgoal2":    
        dataframe["totgoal2"] = np.where( dataframe["totgoal"]==2, 1, 0)
    if label_name =="totgoal3":    
        dataframe["totgoal3"] = np.where( dataframe["totgoal"]==3, 1, 0)
    if label_name =="totgoal4":    
        dataframe["totgoal4"] = np.where( dataframe["totgoal"]==4, 1, 0)
    if label_name =="totgoal5":    
        dataframe["totgoal5"] = np.where( dataframe["totgoal"]==5, 1, 0)

    
    if label_name =="Team1Win":
        dataframe["Team1Win"] = np.where( dataframe["FTR"]==1, 1, 0)
    if label_name =="NotTeam1Win":
        dataframe["NotTeam1Win"] = np.where( dataframe["FTR"]!=1, 1, 0)
    
    if label_name =="Team2Win":    
        dataframe["Team2Win"] = np.where( dataframe["FTR"]==2, 1, 0)
    if label_name =="NotTeam2Win":    
        dataframe["NotTeam2Win"] = np.where( dataframe["FTR"]!=2, 1, 0)
        
    if label_name =="TeamNWin":
        dataframe["TeamNWin"] = np.where( dataframe["FTR"]==0, 1, 0)
    if label_name =="NotTeamNWin":
        dataframe["NotTeamNWin"] = np.where( dataframe["FTR"]!=0, 1, 0)

    if label_name =="Ov05":
        dataframe["Ov05"] = np.where( dataframe["totgoal"]>0, 1, 0)
        
    if label_name =="Ov15":
        dataframe["Ov15"] = np.where( dataframe["totgoal"]>1, 1, 0)
        
    if label_name =="Ov25":
        dataframe["Ov25"] = np.where( dataframe["totgoal"]>2, 1, 0)
        
    if label_name =="Ov35":
        dataframe["Ov35"] = np.where( dataframe["totgoal"]>3, 1, 0)
        
    if label_name =="Ov45":
        dataframe["Ov45"] = np.where( dataframe["totgoal"]>4, 1, 0)
        
    if label_name =="LDEM":
        dataframe["LDEM"] = np.where( dataframe["FTHG"]>0 , np.where( dataframe["FTAG"]>0 , 1, 0), 0)

    ##Column Groups
    #---
    BetCols = ['BW1', 'BWN', 'BW2',
        'BOver1_5goals', 'BUnder1_5goals','BOver2_5goals', 'BUnder2_5goals','BOver3_5goals', 'BUnder3_5goals',
        'BLDEM_YES', 'BLDEM_NO']
    
    PlayersRatingcols = [
        'HTPlayerGScore',
        'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
        'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
        'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
        'ATPlayerGScore',
        'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
        'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
        'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score'
    ]
    for col in PlayersRatingcols:
        dataframeInput[col] = dataframeInput.apply(ratingRanges_v3, args=[col], axis=1); 
        
    PlayersIDcols = [
        'HTPlayerG',
        'HTPlayerD1','HTPlayerD2','HTPlayerD3','HTPlayerD4','HTPlayerD5','HTPlayerD6',
        'HTPlayerM1','HTPlayerM2','HTPlayerM3','HTPlayerM4','HTPlayerM5','HTPlayerM6','HTPlayerM7',
        'HTPlayerF1','HTPlayerF2','HTPlayerF3','HTPlayerF4',
        'ATPlayerG',
        'ATPlayerD1','ATPlayerD2','ATPlayerD3','ATPlayerD4','ATPlayerD5','ATPlayerD6',
        'ATPlayerM1','ATPlayerM2','ATPlayerM3','ATPlayerM4','ATPlayerM5','ATPlayerM6','ATPlayerM7',
        'ATPlayerF1','ATPlayerF2','ATPlayerF3','ATPlayerF4'
    ]
    
    TeamRatingcols=['HTPlayerDScore','HTPlayerMScore','HTPlayerFScore','ATPlayerDScore','ATPlayerMScore','ATPlayerFScore']
    
    #Probablity new columns
    dataframe["DiffWin12"] = dataframe["W1"]-dataframe["W2"]
    dataframe["DiffWin1N"] = dataframe["W1"]-dataframe["WN"]
    dataframe["DiffWin2N"] = dataframe["W2"]-dataframe["WN"]

    
    #Teams Scores new columns
    dataframe[PlayersRatingcols] = dataframe[PlayersRatingcols].replace({0.0:np.nan})
    
    dataframe['HTPlayerDScore'] = dataframe[['HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score']].mean(axis=1)
    dataframe['HTPlayerMScore'] = dataframe[['HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score']].mean(axis=1)
    dataframe['HTPlayerFScore'] = dataframe[['HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score']].mean(axis=1)

    dataframe['ATPlayerDScore'] = dataframe[['ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score']].mean(axis=1)
    dataframe['ATPlayerMScore'] = dataframe[['ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score']].mean(axis=1)
    dataframe['ATPlayerFScore'] = dataframe[['ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score']].mean(axis=1)

    dataframe[PlayersRatingcols] = dataframe[PlayersRatingcols].replace({np.nan:0.0})
    
    ## Add League avg success on first half

    if HT==False:
      ## download halftimeMean 
      tmppaths="https://web-concepts.fr/soccer/win-pronos/pronos-R-data-dev.php?action=halftimeMean"
      
      df_halftime =[]
      df_halftime = pd.read_csv(tmppaths,sep=",", encoding = "ISO-8859-1",index_col=False,low_memory=False)
      
      dataframe['homeWin_halftime'] = dataframe['league_id'].map(df_halftime.set_index('league_id')['homeWin_halftime'])
      dataframe['awayWin_halftime'] = dataframe['league_id'].map(df_halftime.set_index('league_id')['awayWin_halftime'])
    

    # based on provider's predictions 
    dataframe["pred_match_winner_1"] = np.where(  dataframe["pred_match_winner"]=="1", 1, 0)
    dataframe["pred_match_winner_2"] = np.where(  dataframe["pred_match_winner"]=="2", 1, 0)
    dataframe["pred_match_winner_N"] = np.where(  dataframe["pred_match_winner"]=="N", 1, 0)
    
    dataframe["pred_match_winner_1N"] = np.where(  dataframe["pred_match_winner"]=="1 N", 1, 0)
    dataframe["pred_match_winner_N2"] = np.where(  dataframe["pred_match_winner"]=="N 2", 1, 0)
    
    dataframe["pred_goals_home"] = np.where( dataframe["pred_goals_home"]<0, dataframe["pred_goals_home"]+4, 0)
    dataframe["pred_goals_away"] = np.where( dataframe["pred_goals_away"]<0, dataframe["pred_goals_away"]+4, 0)

    # Remove incomplete or bad values

    dataframe = dataframe[dataframe["season"] >2015]
    dataframe = dataframe[dataframe["round"] >2]
    
    #display(dataframe.shape)
    dataframe = dataframe[dataframe.BW1 >0]
    dataframe = dataframe[dataframe.BW2 >0]
    dataframe = dataframe[dataframe.BWN >0]
    
    #display(dataframe.shape)
    
    dataframe = dataframe[dataframe["W1"].between(0, 1)]
    dataframe = dataframe[dataframe["W2"].between(0, 1)]
    dataframe = dataframe[dataframe["WN"].between(0, 1)]
    dataframe = dataframe[dataframe["LDEM_YES"].between(0, 1)]
    #dataframe = dataframe[dataframe["Over0_5goals"].between(0, 1)]
    dataframe = dataframe[dataframe["Over1_5goals"].between(0, 1)]
    dataframe = dataframe[dataframe["Over2_5goals"].between(0, 1)]
    dataframe = dataframe[dataframe["Over3_5goals"].between(0, 1)]
    dataframe = dataframe[dataframe["Over4_5goals"].between(0, 1)]
    
    #display(dataframe.shape)
    ##---- Data Normalization 

    ## per week
    #Normalize Sum Goals scored and conceded per week
    dataframe['HTGS'] = dataframe['HTGS']/dataframe['round']
    dataframe['HTGC'] = dataframe['HTGC']/dataframe['round']
    dataframe['HTP']  = dataframe['HTP'] /dataframe['round']

    dataframe['ATGS'] = dataframe['ATGS']/dataframe['round']
    dataframe['ATGC'] = dataframe['ATGC']/dataframe['round']
    dataframe['ATP']  = dataframe['ATP'] /dataframe['round']

    # Get Goal Difference per week
    dataframe['HTGD'] = dataframe['HTGS'] - dataframe['HTGC']
    dataframe['ATGD'] = dataframe['ATGS'] - dataframe['ATGC']
    
    ## Scalers
    #Normalize Odds
    minmax_scaler = MinMaxScaler()
    std_scaler    = StandardScaler()
    rbst_scaler   = RobustScaler()
    
    #scaled_df = minmax_scaler.fit_transform(dataframe[BetCols])
    #scaled_df = pd.DataFrame(scaled_df, columns=BetCols)
    #dataframe[BetCols] = scaled_df

    #Normalize Rating columns
    RatingCol = PlayersRatingcols + TeamRatingcols
    #scaled_rating_df = minmax_scaler.fit_transform(dataframe[RatingCol])
    #scaled_rating_df = pd.DataFrame(scaled_rating_df, columns=RatingCol)
    #dataframe[RatingCol] = scaled_rating_df
    
    othercol=['round','HTP','HTGS','HTGC','HTF','ATP','ATGS','ATGC','ATF']
    #scaled_other_df = minmax_scaler.fit_transform(dataframe[othercol])
    #scaled_other_df = pd.DataFrame(scaled_other_df, columns=othercol)
    #dataframe[othercol] = scaled_other_df
    
    
    #Fill NANs to 0
    dataframe = dataframe.fillna(0)

    ##----

    dataframe[PlayersIDcols] = dataframe[PlayersIDcols].replace({'0':"IDnull", 0:"IDnull","0":"IDnull"})
    dataframe[PlayersIDcols] = dataframe[PlayersIDcols].replace({np.nan:"IDnull"})
    
    #exclude missing player data
    dataframe=dataframe[dataframe["HTPlayerG"] !="IDnull"]
    #dataframe=dataframe[dataframe["HTPlayerD1"]!="IDnull"]
    #dataframe=dataframe[dataframe["HTPlayerM1"]!="IDnull"]
    #dataframe=dataframe[dataframe["HTPlayerF1"]!="IDnull"]  
    
    dataframe=dataframe[dataframe["ATPlayerG"] !="IDnull"]
    #dataframe=dataframe[dataframe["ATPlayerD1"]!="IDnull"]
    #dataframe=dataframe[dataframe["ATPlayerM1"]!="IDnull"]
    #dataframe=dataframe[dataframe["ATPlayerF1"]!="IDnull"] 
    
    ##----
    #Drop hiddens columns
    dataframe = dataframe.drop(['event_date','goalDiff','goalsHomeTeam','goalsAwayTeam',
                                'FTHG','FTAG','pred_match_winner' ], axis=1)  
    
    
    
    return dataframe

def predict_prepareDataForML(dataframeInput,label_name,HT=False):
    dataframe=dataframeInput.copy()
    
    dataframe=predict_prepareDataForNN(dataframe,label_name,HT)
    features_categorical=[
            'region_name', 
            'country_name', 
            'league_name', 
            'homeTeam', 
            'awayTeam',

            'HTPlayerG',
            'HTPlayerD1','HTPlayerD2','HTPlayerD3','HTPlayerD4','HTPlayerD5','HTPlayerD6',
            'HTPlayerM1','HTPlayerM2','HTPlayerM3','HTPlayerM4','HTPlayerM5','HTPlayerM6','HTPlayerM7',
            'HTPlayerF1','HTPlayerF2','HTPlayerF3','HTPlayerF4',

            'ATPlayerG',
            'ATPlayerD1','ATPlayerD2','ATPlayerD3','ATPlayerD4','ATPlayerD5','ATPlayerD6',
            'ATPlayerM1','ATPlayerM2','ATPlayerM3','ATPlayerM4','ATPlayerM5','ATPlayerM6','ATPlayerM7',
            'ATPlayerF1','ATPlayerF2','ATPlayerF3','ATPlayerF4'
        ]
    features = dataframe[features_categorical]
    
    
    le = LabelEncoder()
    for cat in features_categorical:    
         dataframe[cat]= le.fit_transform(dataframe[cat]) 
    #Remove unnecessary columns
    #dataframe=dataframe.drop(features_categorical, axis=1)
    #dataframe=dataframe.drop([label_name], axis=1)

    return dataframe

def predictNNModel( label_name,  predict_dataframeInput,   trainingDataframe  ):
    pd.set_option('display.max_columns', None)

    #--
    ##convert Predict Dataframe to datasets
    predict_df  = []
    predict_df  = predict_prepareDataForNN(predict_dataframeInput,label_name)
    predict_df = predict_df.reset_index(drop=True)

    #--
    feature_columns= setTrainingFeatureColumns(label_name,trainingDataframe)

    #--
    reset_random_seeds()
    
    #--
    #print("test")
    #display(predict_df.head())
    #display(predict_df.shape)
    
    if predict_df.empty == True:
        print('DataFrame after ', label_name,' cleanup is empty ')
        return null
    else:
        #print('DataFrame after cleanup is OK')
    
        learning_rate = 0.000004539992976248485
        mt            = "model_large_drop"
        decay         = "exponential"    #"step" #time
        epochs        = 100; 
        regul=0.00001
        activation_func='relu' #relu #sigmoid
        #feature_columns,train_df,valid_df= formatData(trainingDataframe,label_name,1)

        #-- hyperparameters
        if label_name=="FTR":
            n_class = 3 ; batch_size = 226 ; momentum = 0.92 ; dropout=0.092 ; mt = "model_large_L2Regul" ; regul=0.0000001
            
            
        elif label_name=="Team1Win" :
            n_class = 1 ; batch_size = 226 ; momentum = 0.85 ; dropout=0.089 ; mt = "model_large_L2Regul" ; regul=0.00000003 
            #feature_columns,train_df,valid_df= formatData(dataframeInput,label_name,0.85)
        elif label_name=="NotTeam1Win" :
            n_class = 1 ; batch_size = 226 ; momentum = 0.85 ; dropout=0.089 ; mt = "model_large_L2Regul" ; regul=0.00000003 
            #feature_columns,train_df,valid_df= formatData(dataframeInput,label_name,0.85)
            
            
        elif label_name=="Team2Win":
            n_class = 1 ; batch_size = 226 ; momentum = 0.85 ; dropout=0.097 ; mt = "model_large_L2Regul" ; regul=0.00000005 
        elif label_name=="NotTeam2Win":
            n_class = 1 ; batch_size = 226 ; momentum = 0.85 ; dropout=0.097 ; mt = "model_large_L2Regul" ; regul=0.000002
            
            
        elif label_name=="TeamNWin":
            n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.092 ; mt = "model_large_L2Regul" ; regul=0.00000003 
            learning_rate = 0.0000003 ; decay="time";
            
            
        elif label_name=="Ov05":
            n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.05 ; mt = "model_large_L2Regul" ; regul=0.00002 ; 
            mt = "model_large_L2Regul_drop_BatchNormalization" ; 
            learning_rate = 0.000001 ; decay="time" ; epochs = 150; 
            
        elif label_name=="Ov15":
            n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.000008 ; 
            mt = "model_large_L2Regul_drop_BatchNormalization" ; 
            learning_rate = 0.000002 ; decay="time" ; #epochs = 250; 
            #feature_columns,train_df,valid_df= formatData(dataframeInput,label_name,0.8);
            
        elif label_name=="Ov25":
            n_class = 1 ; batch_size = 226 ; momentum = 0.87 ; dropout=0.15 ; mt = "model_large_L2Regul" ; regul=0.0000002 

        elif label_name=="Ov35":
            n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.3 ; mt = "model_large_L2Regul" ; regul=0.000008 ; 
            mt = "model_large_L2Regul_drop_BatchNormalization" ; 
            learning_rate = 0.000002 ; decay="time" ; #epochs = 250; 
            #feature_columns,train_df,valid_df= formatData(dataframeInput,label_name,0.8) ;
            
        elif label_name=="Ov45":
            n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.00008 ; 
            mt = "model_large_L2Regul_drop_BatchNormalization" ; 
            learning_rate = 0.000002 ; decay="time" ; #epochs = 250; 
            #feature_columns,train_df,valid_df= formatData(dataframeInput,label_name,0.8);
            
        elif label_name=="LDEM":
            n_class = 1 ; batch_size = 226 ; momentum = 0.80 ; dropout=0.04 ; mt = "model_large_drop" ; regul=0.00000008
            #feature_columns,train_df,valid_df= formatData(dataframeInput,label_name,0.75);
            
        elif label_name=="totgoal0":
            n_class = 1 ; batch_size = 226 ; momentum = 0.80 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.000008
            mt = "model_large_L2Regul_drop_BatchNormalization" ; 
            learning_rate = 0.00003 ; decay="time" #; epochs = 200; 

        elif label_name=="totgoal1":
            n_class = 1 ; batch_size = 226 ; momentum = 0.80 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.000008
            mt = "model_large_L2Regul_drop_BatchNormalization" ; 
            learning_rate = 0.00004 ; decay="time" #; epochs = 200; 

        elif label_name=="totgoal2":
            n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.00001 ; 
            mt = "model_large_L2Regul_drop_BatchNormalization" ; 
            learning_rate = 0.000002 ; decay="time" ; #epochs = 250; 
            #feature_columns,train_df,valid_df= formatData(dataframeInput,label_name,0.8);

        elif label_name=="totgoal3":
            n_class = 1 ; batch_size = 226 ; momentum = 0.80 ; dropout=0.4 ; mt = "model_large_L2Regul" ; regul=0.0000001
            mt = "model_large_L2Regul_drop_BatchNormalization" ; 
            learning_rate = 0.00004 ; decay="time" #; epochs = 200; 


        elif label_name=="totgoal4":
            n_class = 1 ; batch_size = 226 ; momentum = 0.80 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.000008
            mt = "model_large_L2Regul_drop_BatchNormalization" ; 
            learning_rate = 0.00004 ; decay="time" #; epochs = 200; 

        elif label_name=="totgoal5":
            n_class = 1 ; batch_size = 226 ; momentum = 0.95 ; dropout=0.4 ; mt = "model_large_drop" ; regul=0.0000004
            mt = "model_large_L2Regul_drop_BatchNormalization" ; 
            learning_rate = 0.00002 ; decay="time" ; epochs = 48 ; 
            #feature_columns,train_df,valid_df= formatData(dataframeInput,label_name,0.82);
        #----
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum = momentum)

        #----
        ## Create the DNN model
        model = CreateModel(feature_columns,n_class,optimizer = optimizer,selected_model=mt,dropout=dropout,regul=regul)

        ### Load the previously saved weights
        checkpoint_path = "trainingModels/model-cp-"+label_name+"-v4.ckpt"
        model.load_weights(checkpoint_path).expect_partial()

        #convert dataframe to datasheet
        #display(predict_df.head())

        #dataframe = predict_df.drop(['totgoal'] , axis=1)    
        #if label_name !="FTR":
        #    dataframe = dataframe.drop(['FTR'] , axis=1)

        predict_ds = df_to_dataset(predict_df, label_name, shuffle=False, batch_size=batch_size)
        
        ##Generate predictions
        
        predictions = []
        prediction  = model.predict(predict_ds)

        if(label_name=="FTR"):
          predictions = pd.DataFrame( data = prediction, columns=(['D_NN_Pred', 'W_NN_Pred', 'L_NN_Pred']) )
          predictions = predictions.reset_index(drop=True)

          predictions_df = pd.concat(
              [predict_df['fixture_id'], predict_df['FTR'], predict_df['totgoal'], predictions['W_NN_Pred'] ,predictions['D_NN_Pred'] ,predictions['L_NN_Pred']]
              , axis=1)

        elif(label_name=="LDEM"):
          newlabelName=label_name+'_NN_Pred'
          predictions = pd.DataFrame( data = prediction, columns=([newlabelName]) )
          #predictions.rename(columns={label_name: newlabelName})
          #display(predictions.head())
          predictions_df = pd.concat(
              [ predict_df['fixture_id'],predict_df[label_name], predictions ]
              , axis=1)

        elif(label_name=="totgoal0" or label_name=="Ov05"):
          newlabelName=label_name+'_NN_Pred'
          predictions = pd.DataFrame( data = prediction, columns=([newlabelName]) )
          #predictions.rename(columns={label_name: newlabelName})
          #display(predictions.head())
          predictions_df = pd.concat(
              [ predict_df['fixture_id'],predict_df['totgoal'],predict_df[label_name], predictions ]
              , axis=1)

        else:
          newlabelName=label_name+'_NN_Pred'
          predictions = pd.DataFrame( data = prediction, columns=([newlabelName]) )
          #predictions.rename(columns={label_name: newlabelName})
          #display(predictions.head())
          predictions_df = pd.concat(
              [ predict_df[label_name], predictions ]
              , axis=1)

          #display(predictions_df.head())

        #----
        return predictions_df.reset_index(drop=True)


def predictMLModel(label_name, predict_dataframeInput, HT=False):
    import pickle
    reset_random_seeds()
    pd.set_option('display.max_columns', None)

    predict_df = []
    predict_df = predict_prepareDataForML(predict_dataframeInput,label_name,HT)
    predict_df = predict_df.drop([label_name],axis=1)

    if(label_name!="totgoal" and "totgoal" in predict_df.columns): predict_df = predict_df.drop(["totgoal"],axis=1)
    if(label_name!="FTR" and "FTR" in predict_df.columns)        : predict_df = predict_df.drop(["FTR"],axis=1)

    #display(predict_df.shape)
    #display(predict_df.head())

    version ="-v4"
    if HT : version ="-HT-v4"

    #File name definition
    directory='trainingModels/model-cp-RFC-'
    filename        = directory + label_name  + version + '.sav'
    encoderfilename = directory +'encoder-'  + label_name + version + '.npy'

    
    if predict_df.empty == True:
        print('DataFrame after ', label_name,' cleanup is empty ')
        return null
    else:
        
        #load model
        load_model = pickle.load(open(filename, 'rb'))

        #Generate prediction
        #prediction = load_model.predict(predict_df)
        prediction = load_model.predict_proba(predict_df)

        #Decode prediction
        labl_enc = LabelEncoder()
        labl_enc.classes_ = np.load(encoderfilename)
        #predict_df        = labl_enc.inverse_transform(prediction)

        #Make dataframe

        if(label_name=="FTR"):
          predictions_df = pd.DataFrame( data = prediction, columns=(['D_ML_Pred', 'W_ML_Pred', 'L_ML_Pred']) )
        else:
          newlabelName=label_name+'_ML_Pred'
          predictions_df = pd.DataFrame(data = prediction, columns=([newlabelName]))
        #predictions_df = predictions_df.reset_index(drop=True)
        
        #display(pd.DataFrame({
        #    'Nextfixture_id':predict_df['fixture_id']
        #    ,'Predicted': predictions_df
        #}).head())

        #----
        return predictions_df



def predictMLModel_lab( predict_dataframeInput,label_name,HT=False ):
    import pickle
    reset_random_seeds()
    predict_input_df = []

    predict_input_df = predict_prepareDataForML(predict_dataframeInput,label_name,HT)
    dfpr = predict_input_df.drop(["FTR","totgoal"], axis=1)
    #upsampling training dataset
    features_categorical=[
        'region_name', 
        'country_name', 
        'league_name', 
        'homeTeam', 
        'awayTeam',

        'HTPlayerG',
        'HTPlayerD1','HTPlayerD2','HTPlayerD3','HTPlayerD4','HTPlayerD5','HTPlayerD6',
        'HTPlayerM1','HTPlayerM2','HTPlayerM3','HTPlayerM4','HTPlayerM5','HTPlayerM6','HTPlayerM7',
        'HTPlayerF1','HTPlayerF2','HTPlayerF3','HTPlayerF4',

        'ATPlayerG',
        'ATPlayerD1','ATPlayerD2','ATPlayerD3','ATPlayerD4','ATPlayerD5','ATPlayerD6',
        'ATPlayerM1','ATPlayerM2','ATPlayerM3','ATPlayerM4','ATPlayerM5','ATPlayerM6','ATPlayerM7',
        'ATPlayerF1','ATPlayerF2','ATPlayerF3','ATPlayerF4'
    ]
    features = dfpr[features_categorical]
    
    
    
    le = LabelEncoder()
    for cat in features_categorical:    
         dfpr[cat]= le.fit_transform(dfpr[cat]) 

    #display(predict_input_df.shape)
    #display(dfpr.head())
    
    pd.set_option('display.max_columns', None)


    #File name definition
    version ="-v4"
    if HT : version ="-HT-v4"

    directory='trainingModels/model-cp-RFC-'
    filename        = directory + label_name  + version + '-v4.sav'
    encoderfilename = directory +'encoder-'  + label_name + version + '.npy'

    
    if predict_input_df.empty == True:
        print('DataFrame after ', label_name,' cleanup is empty ')
        return null
    else:

        #load model
        load_model = pickle.load(open(filename, 'rb'))
        
        #Make prediction
        if(label_name=="FTR" or label_name=="totgoal"):
          prediction = load_model.predict(dfpr)
        else:
          dfpr = dfpr.drop([label_name], axis=1)
          prediction = load_model.predict(dfpr)

        #Decode Prediction
        labl_enc = LabelEncoder()
        labl_enc.classes_ = np.load(encoderfilename)
        predict_df        = labl_enc.inverse_transform(prediction)
        predict_df = pd.DataFrame( data = predict_df, columns=([label_name]) )
        predict_df = predict_df.reset_index(drop=True)

        #Concat with fixture _Id, FTR_ref, totgoal_ref info 
        if(label_name=="FTR"):
          predictions_df                 = predict_input_df[["fixture_id"]]
          predictions_df["FTR_ref"]      = predict_input_df[["FTR"]]
          predictions_df["totgoal_ref"]  = predict_input_df[["totgoal"]]
          predictions_df = predictions_df.reset_index(drop=True)
          predictions = pd.concat([predictions_df,predict_df], axis=1)
        else:
          predictions=predict_df
        return predictions

def setTrainingFeatureColumns(label_name,trainingDataframe):
    learning_rate = 0.000004539992976248485
    #make dataframe
    dataframe=prepareDataForFeatureCol(trainingDataframe,label_name)
    #print("Preparing Data... ",dataframe.shape)

    #Set the feature columns
    feature_columns = setFeatureColumns(dataframe)
    #print("end building feature columns")
    return feature_columns

def prepareDataForFeatureCol(dataframeInput,label_name,HT=False):
    
    dataframe=dataframeInput.copy()
    
    #Remove unnecessary columns
    dataframe=dataframe.drop([
        "homeUpdated",
        "awayUpdated",
        "statusShort",
        "status"
    ], axis=1)
    
    #Format conversions
    dataframe["event_date"]   = pd.to_datetime(dataframe["event_date"], format='%Y-%m-%d')
    dataframe["LDEM_YES"]     = pd.to_numeric(dataframe["LDEM_YES"], errors='coerce')
    dataframe["Over2_5goals"] = pd.to_numeric(dataframe["Over2_5goals"], errors='coerce')
    dataframe["round"]        = dataframe["round"].fillna(0.0).astype(int)
    
    #Add New Data to the dataset
    dataframe["totgoal"]  = dataframe['goalsHomeTeam']+dataframe['goalsAwayTeam']
    dataframe["goalDiff"] = dataframe['goalsHomeTeam']-dataframe['goalsAwayTeam']
    dataframe["FTHG"]     = dataframe['goalsHomeTeam']
    dataframe["FTAG"]     = dataframe['goalsAwayTeam']
    #dataframe["Season"]=dataframe['season']
    
    #CREATE BINARY OUTPUTS
    
    #FixtureTimeResult values =>  retrun 1 if team1; 2 if team2 ; 0 if draw ; else -1
    dataframe['FTR'] = dataframe.apply(ftrRule, axis=1)
    
    
    if label_name =="Team1Win":    dataframe["Team1Win"]    = np.where( dataframe["FTR"]==1, 1, 0)
    if label_name =="NotTeam1Win": dataframe["NotTeam1Win"] = np.where( dataframe["FTR"]!=1, 1, 0)
    
    if label_name =="Team2Win":    dataframe["Team2Win"]    = np.where( dataframe["FTR"]==2, 1, 0)
    if label_name =="NotTeam2Win": dataframe["NotTeam2Win"] = np.where( dataframe["FTR"]!=2, 1, 0)
        
    if label_name =="TeamNWin":    dataframe["TeamNWin"]    = np.where( dataframe["FTR"]==0, 1, 0)
    if label_name =="NotTeamNWin": dataframe["NotTeamNWin"] = np.where( dataframe["FTR"]!=0, 1, 0)

    if label_name =="Ov05":        dataframe["Ov05"] = np.where( dataframe["totgoal"]>0, 1, 0)
    if label_name =="Ov15":        dataframe["Ov15"] = np.where( dataframe["totgoal"]>1, 1, 0)
    if label_name =="Ov25":        dataframe["Ov25"] = np.where( dataframe["totgoal"]>2, 1, 0)
    if label_name =="Ov35":        dataframe["Ov35"] = np.where( dataframe["totgoal"]>3, 1, 0)
    if label_name =="Ov45":        dataframe["Ov45"] = np.where( dataframe["totgoal"]>4, 1, 0)
        
    if label_name =="LDEM":        dataframe["LDEM"] = np.where( dataframe["FTHG"]>0 , np.where( dataframe["FTAG"]>0 , 1, 0), 0)

    if label_name =="totgoal0":    dataframe["totgoal0"] = np.where( dataframe["totgoal"]==0, 1, 0)
    if label_name =="totgoal1":    dataframe["totgoal1"] = np.where( dataframe["totgoal"]==1, 1, 0)
    if label_name =="totgoal2":    dataframe["totgoal2"] = np.where( dataframe["totgoal"]==2, 1, 0)
    if label_name =="totgoal3":    dataframe["totgoal3"] = np.where( dataframe["totgoal"]==3, 1, 0)
    if label_name =="totgoal4":    dataframe["totgoal4"] = np.where( dataframe["totgoal"]==4, 1, 0)
    if label_name =="totgoal5":    dataframe["totgoal5"] = np.where( dataframe["totgoal"]==5, 1, 0)


    ##Column Groups
    #---
    BetCols = ['BW1', 'BWN', 'BW2',
        'BOver1_5goals', 'BUnder1_5goals','BOver2_5goals', 'BUnder2_5goals','BOver3_5goals', 'BUnder3_5goals',
        'BLDEM_YES', 'BLDEM_NO']
    
    PlayersRatingcols = [
        'HTPlayerGScore',
        'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
        'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
        'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
        'ATPlayerGScore',
        'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
        'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
        'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score'
    ]
       
    PlayersIDcols = [
        'HTPlayerG',
        'HTPlayerD1','HTPlayerD2','HTPlayerD3','HTPlayerD4','HTPlayerD5','HTPlayerD6',
        'HTPlayerM1','HTPlayerM2','HTPlayerM3','HTPlayerM4','HTPlayerM5','HTPlayerM6','HTPlayerM7',
        'HTPlayerF1','HTPlayerF2','HTPlayerF3','HTPlayerF4',
        'ATPlayerG',
        'ATPlayerD1','ATPlayerD2','ATPlayerD3','ATPlayerD4','ATPlayerD5','ATPlayerD6',
        'ATPlayerM1','ATPlayerM2','ATPlayerM3','ATPlayerM4','ATPlayerM5','ATPlayerM6','ATPlayerM7',
        'ATPlayerF1','ATPlayerF2','ATPlayerF3','ATPlayerF4'
    ]
    
    TeamRatingcols=['HTPlayerDScore','HTPlayerMScore','HTPlayerFScore','ATPlayerDScore','ATPlayerMScore','ATPlayerFScore']
    
    #Probablity new columns
    dataframe["DiffWin12"] = dataframe["W1"]-dataframe["W2"]
    dataframe["DiffWin1N"] = dataframe["W1"]-dataframe["WN"]
    dataframe["DiffWin2N"] = dataframe["W2"]-dataframe["WN"]

    
    #Teams Scores new columns
    dataframe[PlayersRatingcols] = dataframe[PlayersRatingcols].replace({0.0:np.nan})
    
    dataframe['HTPlayerDScore'] = dataframe[['HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score']].mean(axis=1)
    dataframe['HTPlayerMScore'] = dataframe[['HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score']].mean(axis=1)
    dataframe['HTPlayerFScore'] = dataframe[['HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score']].mean(axis=1)

    dataframe['ATPlayerDScore'] = dataframe[['ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score']].mean(axis=1)
    dataframe['ATPlayerMScore'] = dataframe[['ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score']].mean(axis=1)
    dataframe['ATPlayerFScore'] = dataframe[['ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score']].mean(axis=1)

    dataframe[PlayersRatingcols] = dataframe[PlayersRatingcols].replace({np.nan:0.0})
    
    
    
    ## download halftimeMean 

    if HT==False:
      tmppaths="https://web-concepts.fr/soccer/win-pronos/pronos-R-data-dev.php?action=halftimeMean"
      
      df_halftime =[]
      df_halftime = pd.read_csv(tmppaths,sep=",", encoding = "ISO-8859-1",index_col=False,low_memory=False)
      
      dataframe['homeWin_halftime'] = dataframe['league_id'].map(df_halftime.set_index('league_id')['homeWin_halftime'])
      dataframe['awayWin_halftime'] = dataframe['league_id'].map(df_halftime.set_index('league_id')['awayWin_halftime'])
    
    
    # based on provider's predictions 
    dataframe["pred_match_winner_1"] = np.where(  dataframe["pred_match_winner"]=="1", 1, 0)
    dataframe["pred_match_winner_2"] = np.where(  dataframe["pred_match_winner"]=="2", 1, 0)
    dataframe["pred_match_winner_N"] = np.where(  dataframe["pred_match_winner"]=="N", 1, 0)
    
    dataframe["pred_match_winner_1N"] = np.where(  dataframe["pred_match_winner"]=="1 N", 1, 0)
    dataframe["pred_match_winner_N2"] = np.where(  dataframe["pred_match_winner"]=="N 2", 1, 0)
    
    dataframe["pred_goals_home"] = np.where( dataframe["pred_goals_home"]<0, dataframe["pred_goals_home"]+4, 0)
    dataframe["pred_goals_away"] = np.where( dataframe["pred_goals_away"]<0, dataframe["pred_goals_away"]+4, 0)

    
    # Remove incomplete or bad values

    dataframe = dataframe[dataframe["season"] >2015]
    dataframe = dataframe[dataframe["round"] >=2]
    
    dataframe = dataframe[dataframe.BW1 >0]
    dataframe = dataframe[dataframe.BW2 >0]
    dataframe = dataframe[dataframe.BWN >0]
    
    
    dataframe = dataframe[dataframe["W1"].between(0, 1)]
    dataframe = dataframe[dataframe["W2"].between(0, 1)]
    dataframe = dataframe[dataframe["WN"].between(0, 1)]
    dataframe = dataframe[dataframe["LDEM_YES"].between(0, 1)]
    dataframe = dataframe[dataframe["Over0_5goals"].between(0, 1)]
    dataframe = dataframe[dataframe["Over1_5goals"].between(0, 1)]
    dataframe = dataframe[dataframe["Over2_5goals"].between(0, 1)]
    dataframe = dataframe[dataframe["Over3_5goals"].between(0, 1)]
    dataframe = dataframe[dataframe["Over4_5goals"].between(0, 1)]
    
    ##---- Data Normalization
    
    #Normalize Sum Goals scored and conceded per week
    dataframe['HTGS'] = dataframe['HTGS']/dataframe['round']
    dataframe['HTGC'] = dataframe['HTGC']/dataframe['round']
    dataframe['HTP']  = dataframe['HTP'] /dataframe['round']

    dataframe['ATGS'] = dataframe['ATGS']/dataframe['round']
    dataframe['ATGC'] = dataframe['ATGC']/dataframe['round']
    dataframe['ATP']  = dataframe['ATP'] /dataframe['round']

    # Get Goal Difference per week
    dataframe['HTGD'] = dataframe['HTGS'] - dataframe['HTGC']
    dataframe['ATGD'] = dataframe['ATGS'] - dataframe['ATGC']
    
    #Normalize Odds
    minmax_scaler = MinMaxScaler()
    std_scaler    = StandardScaler()
    rbst_scaler   = RobustScaler()
    
    #scaled_df = minmax_scaler.fit_transform(dataframe[BetCols])
    #scaled_df = pd.DataFrame(scaled_df, columns=BetCols)
    #dataframe[BetCols] = scaled_df

    #Normalize Rating columns
    RatingCol = PlayersRatingcols + TeamRatingcols
    #scaled_rating_df = minmax_scaler.fit_transform(dataframe[RatingCol])
    #scaled_rating_df = pd.DataFrame(scaled_rating_df, columns=RatingCol)
    #dataframe[RatingCol] = scaled_rating_df
    
    othercol=['round','HTP','HTGS','HTGC','HTF','ATP','ATGS','ATGC','ATF']
    #scaled_other_df = minmax_scaler.fit_transform(dataframe[othercol])
    #scaled_other_df = pd.DataFrame(scaled_other_df, columns=othercol)
    #dataframe[othercol] = scaled_other_df
    
    
    #Fill NANs to 0
    dataframe = dataframe.fillna(0)

    ##----

    dataframe[PlayersIDcols] = dataframe[PlayersIDcols].replace({'0':"IDnull", 0:"IDnull","0":"IDnull"})
    dataframe[PlayersIDcols] = dataframe[PlayersIDcols].replace({np.nan:"IDnull"})
    
    #exclude missing player data
    dataframe=dataframe[dataframe["HTPlayerG"] !="IDnull"]
    #dataframe=dataframe[dataframe["HTPlayerD1"]!="IDnull"]
    #dataframe=dataframe[dataframe["HTPlayerM1"]!="IDnull"]
    #dataframe=dataframe[dataframe["HTPlayerF1"]!="IDnull"]  
    
    dataframe=dataframe[dataframe["ATPlayerG"] !="IDnull"]
    #dataframe=dataframe[dataframe["ATPlayerD1"]!="IDnull"]
    #dataframe=dataframe[dataframe["ATPlayerM1"]!="IDnull"]
    #dataframe=dataframe[dataframe["ATPlayerF1"]!="IDnull"]  
    
    
    #target variables Encoding: one hot encoding, Label encoding
    #dataframe['event_date'] = pd.Categorical(dataframe['event_date'])
    #dataframe['event_date'] = dataframe.event_date.cat.codes
    
    #le = LabelEncoder() 
    #dataframe['region_name']= le.fit_transform(dataframe['region_name']) 
    #dataframe['country_name']= le.fit_transform(dataframe['country_name']) 
    #dataframe['league_name']= le.fit_transform(dataframe['league_name']) 
    #dataframe['homeTeam']= le.fit_transform(dataframe['homeTeam']) 
    #dataframe['awayTeam']= le.fit_transform(dataframe['awayTeam']) 
    #for playerid in PlayersIDcols:    
    #     dataframe[playerid]= le.fit_transform(dataframe[playerid]) 
    
    
    ##----
    #Country filtering
    #dataframe=dataframe[dataframe["country_name"]!="Denmark"]
    
    
    ##----
    #Drop hiddens columns
    dataframe = dataframe.drop(['event_date','totgoal','goalDiff','goalsHomeTeam','goalsAwayTeam',
                                'FTHG','FTAG','pred_match_winner' ], axis=1)  
    
    
    if label_name !="FTR" and 'FTR' in dataframe.columns:
        dataframe = dataframe.drop(['FTR'] , axis=1)  
    
    ##----
    #Mélange les données
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    
    return dataframe


def prd_rules(x):
    if (
        x['Team1Win'] > 0.51
        and x['NotTeam2Win'] > 0.51
        and x['W'] > x['L']
        and x['W'] > x['D']
        and x['TeamNWin']==0
    ):
        return "1"
    elif (
        x['Team2Win'] > 0.51 
        and x['NotTeam1Win'] > 0.51 
        and x['L'] > x['W']
        and x['L'] > x['D']
        and x['TeamNWin']==0
    ) :
        return "2"
    elif (
        x['NotTeam1Win'] > 0.55
        and x['NotTeam2Win'] > 0.55
        and x['D'] > 0.51
        and x['D'] > x['W']
        and x['D'] > x['L']
        and x['TeamNWin']==1
    ) :
        return "N"
    else:
        return " ? "

 
def prd_TOTGOAL_OverRules(x):
    result=""
    if (x['Ov05'] > 1 ):
        result= "Over 1.5"
    if (x['Ov15'] > 0  and x['Ov05'] > 0):
        result= "Over 1.5"
    if (x['Ov25'] > 0 and x['Ov15'] > 0 and x['Ov05'] > 0):
        result= "Over 2.5"
    if (x['Ov35'] > 0  and x['Ov25'] > 0 and x['Ov15'] > 0 and x['Ov05'] > 0):
        result= "Over 3.5"
    if (x['Ov45'] > 0 and x['Ov35'] > 0 and x['Ov25'] > 0 and x['Ov15'] > 0 and x['Ov05'] > 0):
        result= "Over 4.5"
    return result

def prd_TOTGOAL_UnderRules(x):
    result=""

    if (x['Ov45'] < 1):
        result= "Under 4.5"
    if (x['Ov45'] < 1 and x['Ov35'] < 1):
        result= "Under 3.5"
    if (x['Ov45'] < 1 and x['Ov35'] < 1 and x['Ov25'] <1 ):
        result= "Under 2.5"
    if (x['Ov45'] < 1 and x['Ov35'] < 1 and x['Ov25'] <1 and x['Ov15'] < 0 ):
        result= "Under 1.5"
    if (x['Ov45'] < 1 and x['Ov35'] < 1 and x['Ov25'] <1 and x['Ov15'] < 0 and x['Ov05'] < 0 ):
        result= "under 0.5"
    return result
       
def prd_OU35_rules(x):
    if (x['Ov35Win'] > 0.5 ):
        return "Over"
    else:
        return ""
def prd_OU15_rules(x):
    if x['Ov15Win'] > 0.5  :
        return "Over"
    else:
        return ""
    
    
def prd_T1_rules(x):
    if (x['Team1Win'] > 0.5  
        and x['NotTeam2Win'] > 0.51 
        and x['W'] > x['L'] 
        and x['TeamNWin']==0
       ):
        return "1"
    else:
        return ""
def prd_T2_rules(x):
    if (x['Team2Win'] > 0.5  
        and x['NotTeam1Win'] > 0.51 
        and x['L'] > x['W']
        and x['TeamNWin']==0
       ):
        return "2"
    else:
        return ""
def prd_TN_rules(x):
    if (x['NotTeam1Win'] > 0.60  and x['NotTeam2Win'] > 0.60
        and x['D'] > x['W']
        and x['D'] > x['L']
        and x['TeamNWin']==1
       ):
        return "N"
    else:
        return ""
    
def prd_1N2_rules(x):
    if   x['W'] > x['D'] and x['W'] > x['L'] and x['W'] > 0.5 and x['TeamNWin'] ==0 : return "1"
    elif x['L'] > x['D'] and x['L'] > x['W'] and x['L'] > 0.5and x['TeamNWin'] ==0 : return "2"
    elif x['D'] > x['W'] and x['W'] > x['L'] and (50 - x['D']) > (x['D'] - x['W']) and (x['W'] - x['L'])>6 and x['TeamNWin'] ==1 :  return "1N"
    elif x['D'] > x['L'] and x['L'] > x['W'] and (50 - x['D']) > (x['D'] - x['L']) and (x['L'] - x['W'])>6 and x['TeamNWin'] ==1 :  return "N2"
    elif x['D'] > x['W'] and x['D'] > x['L'] and x['NotTeam1Win'] > 0.60 and x['NotTeam2Win'] > 0.60 and x['TeamNWin'] ==1  :  return "N"
    else:
        return ""

def controlPlayers(df):
    testPlayers = True
    invalidPlayer = ""
    df["playerErrors"] = 0

    for index, row in df.iterrows():
        PlayersRatingcols = [
            'HTPlayerGScore',
            'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
            'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
            'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
            'ATPlayerGScore',
            'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
            'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
            'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score'
        ]    
        countErrors=0
        countplayers=0
        for col in PlayersRatingcols:
            #print(row[col])
            if( float(row[col]).is_integer() and testPlayers): 
                testPlayers=True;
                countplayers+=1
            else: 
                testPlayers=False ;

            if( float(row[col]).is_integer() ==False): 
                invalidPlayer = invalidPlayer + ", " + col;
                row[col]=0 ; 
                countErrors+=1
                #print("error ",col, " on: ",row[col])
        #print("=>match players (",countplayers,") errors found:",countErrors)
        df.at[index, "playerErrors"] = countErrors
        #print(df.iloc[index]["playerErrors"])
    return df


def testPlayers(df):
    testPlayers=True
    invalidPlayer=""
    PlayersRatingcols = [
        'HTPlayerGScore',
        'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
        'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
        'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
        'ATPlayerGScore',
        'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
        'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
        'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score'
    ]    
    countErrors=0
    for col in PlayersRatingcols:
        if( float(df.iloc[0][col]).is_integer() and testPlayers): 
            testPlayers=True
        else: 
            testPlayers=False ;
            
        if( float(df.iloc[0][col]).is_integer() ==False): 
            invalidPlayer = invalidPlayer + ", " + col;df.iloc[0][col]=0 ; countErrors+=1
    
    
    display(HTML("<div style='text-align:center;width:500px'>G : "+ str(df.iloc[0]['HTPlayerGScore'])+"</div>" )) 
    
    strDisplay=""
    i=0
    Playerscols = ['HTPlayerD1','HTPlayerD2','HTPlayerD3','HTPlayerD4','HTPlayerD5','HTPlayerD6']
    for pl in Playerscols:
        i+=1
        if(pd.notna(df.iloc[0][pl])): 
            strDisplay += "<span style='width:80px;text-align:center;display:inline-block'>D" + str(i) + " : " + str(df.iloc[0][pl+'Score'])+"</span>"
    display(HTML("<div style='text-align:center;width:500px'>"+strDisplay+"</div>" ))
    
    
    strDisplay=""
    i=0
    Playerscols = ['HTPlayerM1','HTPlayerM2','HTPlayerM3','HTPlayerM4','HTPlayerM5','HTPlayerM6','HTPlayerM7']
    for pl in Playerscols:
        i+=1
        if(pd.notna(df.iloc[0][pl])):
            strDisplay += '<span style="width:80px;text-align:center;display:inline-block">M' + str(i) + " : " + str(df.iloc[0][pl+'Score'])+"</span>"
    display(HTML("<div style='text-align:center;width:500px'>"+strDisplay+"</div>" ))
    
    
    strDisplay=""
    i=0
    Playerscols = ['HTPlayerF1','HTPlayerF2','HTPlayerF3','HTPlayerF4']
    for pl in Playerscols:
        i+=1
        if(pd.notna(df.iloc[0][pl])):
            strDisplay += '<span style="width:80px;text-align:center;display:inline-block">F' + str(i) + " : " + str(df.iloc[0][pl+'Score'])+"</span>"
    display(HTML("<div style='text-align:center;width:500px'>"+strDisplay+"</div>" ))
    
    display(HTML("<div style='text-align:center;width:500px'> -------- O -------- </div>" ))
    
    strDisplay=""
    i=0
    Playerscols = ['ATPlayerF1','ATPlayerF2','ATPlayerF3','ATPlayerF4']
    for pl in Playerscols:
        i+=1
        if(pd.notna(df.iloc[0][pl])):
            strDisplay += '<span style="width:80px;text-align:center;display:inline-block">F' + str(i) + " : " + str(df.iloc[0][pl+'Score'])+"</span>"
    display(HTML("<div style='text-align:center;width:500px'>"+strDisplay+"</div>" ))
    
    
    strDisplay=""
    i=0
    Playerscols = ['ATPlayerM1','ATPlayerM2','ATPlayerM3','ATPlayerM4','ATPlayerM5','ATPlayerM6','ATPlayerM7']
    for pl in Playerscols:
        i+=1
        if(pd.notna(df.iloc[0][pl])):
            strDisplay += '<span style="width:80px;text-align:center;display:inline-block">M' + str(i) + " : " + str(df.iloc[0][pl+'Score'])+"</span>"
    display(HTML("<div style='text-align:center;width:500px'>"+strDisplay+"</div>" ))
    
    strDisplay=""
    i=0
    Playerscols = ['ATPlayerD1','ATPlayerD2','ATPlayerD3','ATPlayerD4','ATPlayerD5','ATPlayerD6']
    for pl in Playerscols:
        i+=1
        if(pd.notna(df.iloc[0][pl])):
            strDisplay += '<span style="width:80px;text-align:center;display:inline-block">D' + str(i) + " : " + str(df.iloc[0][pl+'Score'])+"</span>"
    display(HTML("<div style='text-align:center;width:500px'>"+strDisplay+"</div>" ))
    
    display(HTML("<div style='text-align:center;width:500px'>G : "+ str(df.iloc[0]['ATPlayerGScore'])+"</div>" )) 
    
    print("")
    if(testPlayers) : print("     --> Valid Bet: Players rating : OK")
    elif( 
      float(df.iloc[0]['HTPlayerGScore']).is_integer()==False
      or float(df.iloc[0]['ATPlayerGScore']).is_integer()==False
     ): print("     *****  ERROR : Invalid Goal players !  ",invalidPlayer)
        
    else : print("     *****  WARNING : Invalid players !  ",invalidPlayer)
    
    return countErrors, df

def getPlayersRatings(importplayers=False,date="",offset=0):
    
    if(importplayers) :
        print("")
        print("#### Create the player dataset ...")
        url0="https://web-concepts.fr/soccer/update_fixtures_live.php"
        r0=requests.get(url0) 
        if(r0.status_code==200): print("  -> Update live fixtures : ok")

        url="https://web-concepts.fr/soccer/updateDatasetPlayers.php"
        if(date!="") :
            url="https://web-concepts.fr/soccer/updateDatasetPlayers.php?offset="+str(offset)+"&startDate="+str(date)
        r=requests.get(url) 
        if(r.status_code==200): print("  -> player dataset created: ok")
            
    # Make Player prediction for Dataset
    predictPlayerRating(startDate=date)

#----------------
defaultModelVersion="v4"
def StackFileNames(label_name, version=defaultModelVersion, HT=False):
    
    directory       = 'trainingModels/model-cp-RFC-stack-'
    version ="-" + version
    if HT: version = "-HT-" + version
        
    filename        = directory + label_name + version +'.sav'
    encoderfilename = directory +'encoder-'  + label_name + version +'.npy'
    return filename,encoderfilename

def MLPredict(label_name, df):
    
    filename,encoderfilename= StackFileNames(label_name)
    #-
    load_model = pickle.load(open(filename, 'rb'))
    
    reset_random_seeds()
    prediction  = load_model.predict(df)

    labl_enc = LabelEncoder()
    labl_enc.classes_ = np.load(encoderfilename)

    prediction    = labl_enc.inverse_transform(prediction)
    prediction    = pd.DataFrame(data = prediction, columns=([label_name]))
    
    return prediction

def MLPredictTest(label_name, x_train, y_train, x_test, y_test):
    
    filename,encoderfilename= StackFileNames(label_name)
    
    #Load model
    tmp_load_model = pickle.load(open(filename, 'rb'))

    pred_train  = tmp_load_model.predict(x_train)
    accuracy_score_train = accuracy_score(y_train,pred_train)

    pred_test = tmp_load_model.predict(x_test)
    accuracy_score_test = accuracy_score(y_test,pred_test)

    print("     ... Train - accuracy_score = %1.3f" % accuracy_score_train)
    print("     ... Test  - accuracy_score = %1.3f" % accuracy_score_test)
    print("     ... Test  - classification:\n",classification_report(y_test, pred_test))
    
    return True

def Validate(df):
    df= formatPlayerRatingRange(df)
    tdf = pd.read_csv("dfFeatureColumns-v4.csv", sep="\t", encoding = "ISO-8859-1", index_col=False, low_memory=False)
    
    #Predictions
    FTR_NN_predictions  = predictNNModel( "FTR", df, tdf); print("FTR DNN OK")
    FTR_ML_predictions  = predictMLModel( "FTR", df); print("FTR ML OK")
    
    Stack_FTR_predictions  = ValidationStackModed(FTR_NN_predictions.drop(['totgoal'],axis=1),FTR_ML_predictions,"FTR")

    
    LDEM_NN_predictions    = predictNNModel( "LDEM", df, tdf); print("LDEM DNN OK")
    LDEM_ML_predictions    = predictMLModel( "LDEM", df) ;print("LDEM ML OK")
    
    #Stack Predictions
    Stack_LDEM_predictions = ValidationStackModed(LDEM_NN_predictions.drop(["fixture_id"],axis=1),LDEM_ML_predictions,"LDEM")

    totgoal0_NN_predictions    = predictNNModel( "totgoal0", df, tdf) ; print("totgoal0 DNN OK")
    totgoal0_ML_predictions    = predictMLModel( "totgoal0", df) ; print("totgoal0 ML OK")
    totgoal1_NN_predictions    = predictNNModel( "totgoal1", df, tdf) ; print("totgoal1 DNN OK")
    totgoal1_ML_predictions    = predictMLModel( "totgoal1", df) ; print("1 ML OK")
    totgoal2_NN_predictions    = predictNNModel( "totgoal2", df, tdf) ; print("totgoal2 DNN OK")
    totgoal2_ML_predictions    = predictMLModel( "totgoal2", df) ; print("totgoal2 ML OK")
    totgoal3_NN_predictions    = predictNNModel( "totgoal3", df, tdf) ; print("totgoal3 DNN OK")
    totgoal3_ML_predictions    = predictMLModel( "totgoal3", df) ; print("totgoal3 ML OK")
    totgoal4_NN_predictions    = predictNNModel( "totgoal4", df, tdf) ; print("totgoal4 DNN OK")
    totgoal4_ML_predictions    = predictMLModel( "totgoal4", df) ; print("totgoal4 ML OK")
    totgoal5_NN_predictions    = predictNNModel( "totgoal5", df, tdf) ;print("totgoal5 DNN OK")
    totgoal5_ML_predictions    = predictMLModel( "totgoal5", df) ; print("totgoal5 ML OK")

    #Stack Totalgoal
    Stack_totgoal_df = pd.concat(
              [ totgoal0_NN_predictions["totgoal"]
               , totgoal0_NN_predictions.drop(['totgoal',"totgoal0"],axis=1), totgoal0_ML_predictions
               , totgoal1_NN_predictions.drop(["totgoal1"],axis=1), totgoal1_ML_predictions
               , totgoal2_NN_predictions.drop(["totgoal2"],axis=1), totgoal2_ML_predictions
               , totgoal3_NN_predictions.drop(["totgoal3"],axis=1), totgoal3_ML_predictions
               , totgoal4_NN_predictions.drop(["totgoal4"],axis=1), totgoal4_ML_predictions
               , totgoal5_NN_predictions.drop(["totgoal5"],axis=1)
              ]
              , axis=1)
    
    Stack_totgoal_predictions = ValidationStackModed(Stack_totgoal_df,totgoal5_ML_predictions,"totgoal")


def ValidationStackModed(predictions_NN,predictions_ML,label_name,trainparam=""):
    
    filename,encoderfilename= StackFileNames(label_name)
    
    data    = pd.concat([predictions_NN,predictions_ML] , axis=1, join="outer") #.drop([label_name],axis=1)
    x_valid = data.drop([label_name],axis=1)
    y_valid = data[label_name]

    #display(data)
    #-
    load_model = pickle.load(open(filename, 'rb'))

    reset_random_seeds()
    pred_valid  = load_model.predict(x_valid)


    accuracy_score_valid = accuracy_score(y_valid,pred_valid)

    print("     ... Validation  - accuracy_score = %1.3f" % accuracy_score_valid)
    print("     ... Validation  - classification:\n",classification_report(y_valid, pred_valid))
    
    return True

def PredictStackModeds(predictions_NN,predictions_ML,label_name,trainparam=""):
    
    filename,encoderfilename= StackFileNames(label_name)
    
    data = pd.concat([predictions_NN,predictions_ML] , axis=1, join="outer") #.drop([label_name],axis=1)
    #display(data)
    #-
    load_model = pickle.load(open(filename, 'rb'))
    
    reset_random_seeds()
    prediction  = load_model.predict(data.drop([label_name],axis=1))

    labl_enc = LabelEncoder()
    labl_enc.classes_ = np.load(encoderfilename)

    prediction    = labl_enc.inverse_transform(prediction)
    prediction    = pd.DataFrame(data = prediction, columns=([label_name]))
    
    return prediction


def ratingRanges_v3(x,col):
        if (x[col] > 2.5 and x[col] <= 3.5)  :
            return 3
        elif (x[col] > 3.5 and x[col] <= 4.5)  :
            return 4
        elif (x[col] > 4.5 and x[col] <= 5.5)  :
            return 5
        elif (x[col] > 5.5 and x[col] <= 6.5)  :
            return 6
        elif (x[col] > 6.5 and x[col] <= 7.5)  :
            return 7
        elif (x[col] > 7.5 and x[col] <= 8.5)  :
            return 8
        elif (x[col] > 8.5 and x[col] <= 9.5)  :
            return 9
        elif (x[col] > 9.5)   :
            return 10
        else:
            return 0   

def formatPlayerRatingRange(df):
  PlayersRatingcols = [
      'HTPlayerGScore',
      'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
      'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
      'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
      'ATPlayerGScore',
      'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
      'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
      'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score'
  ]
  for col in PlayersRatingcols:
      df[col] = df.apply(ratingRanges_v3, args=[col], axis=1); 

  return df

