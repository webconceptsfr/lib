




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
    checkpoint_path = "trainingModels/model-cp-"+label_name+"-v3.ckpt"
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
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(128, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(32, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_small_L2Regul = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_large_L2Regul = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_vlarge_L2Regul = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    
    
    #---- BatchNormalization
    model_large_L2Regul_drop_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_large_L2Regul_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_vlarge_L2Regul_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(n_class, activation=activationFn)
    ])
    
    model_large_L1L2Regul_BatchNormalization = tf.keras.Sequential([
      feature_layer,
        layers.BatchNormalization(),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l2(regul)),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1_l2(regul)),
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
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(n_class, activation=activationFn)
    ])
    model_large_L1Regul = tf.keras.Sequential([
      feature_layer,
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1(regul)),
        layers.Dropout(dropout),
      layers.Dense(256, activation=activation_func,
                 kernel_regularizer=regularizers.l1(regul)),
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

def formatData(dataframeInput,label_name,split_ratio):
    #################### PREPARE THE DATA  #####################
    #Train dataframe
    print("############### Training ",label_name," ###############")
    dataframe=prepareData(dataframeInput,label_name)
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

