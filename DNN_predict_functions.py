############################################

import importlib

import libraries 
importlib.reload(libraries)
from libraries import *

import PredictPlayerRating as ppr
importlib.reload(ppr)
headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '\
           'AppleWebKit/537.36 (KHTML, like Gecko) '\
           'Chrome/75.0.3770.80 Safari/537.36'}
#########################################################

defaultModelVersion = "v5"
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
#################### METHODS / FUNCTIONS / DEFINITIONS #####################

def PlotHistory(modelFitHistory,label_name):
		
		dateTimeObj = datetime.now()
		timestampStr = dateTimeObj.strftime("%Y-%m-%d")
		
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		plt.rcParams["figure.figsize"] = (12, 5)
		
		#PLOT Accuracy
		plt.plot(modelFitHistory.history["accuracy"])
		plt.plot(modelFitHistory.history["val_accuracy"])
		
		plt.title('model accuracy for '+label_name+' - '+timestampStr)
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['training','validation'],loc='upper left')
		
		# Show the major grid lines with dark grey lines
		plt.grid(b=True, which='major', color='#666666', linestyle='-')

		# Show the minor grid lines with very faint and almost transparent grey lines
		plt.minorticks_on()
		plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

		plt.show() 
		
		#----
		#PLOT Loss
		plt.plot(modelFitHistory.history["loss"])
		plt.plot(modelFitHistory.history["val_loss"])
		#plt.ylim(0.4, 2)
		
		plt.title('model Loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['training','validation'],loc='upper right')
		
		# Show the major grid lines with dark grey lines
		plt.grid(b=True, which='major', color='#666666', linestyle='-')

		# Show the minor grid lines with very faint and almost transparent grey lines
		plt.minorticks_on()
		plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

		plt.show()
 #----

 #Create an input pipeline using tf.data

def FileNames(label_name, version=defaultModelVersion, model_type="RFC" , HT=False):
		
		directory       = 'trainingModels/model-cp-' + model_type + '-'
		version ="-" + version
		if HT: version = "-HT-" + version
				
		filename        = directory + label_name + version +'.sav'
		encoderfilename = directory +'encoder-'  + label_name + version +'.npy'
		return filename,encoderfilename

def isNotNaN(num):
		return num == num
###################################################

#-------- Deep LEARNING ------
def step_decay(epoch):
	 initial_lrate = 0.000001
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
	 
class LossHistory(keras.callbacks.Callback):
		def on_train_begin(self, logs={}):
			 self.losses = []
			 self.lr = []
 
		def on_epoch_end(self, batch, logs={}):
				self.losses.append(logs.get('loss'))
				self.lr.append((len(self.losses)))
				print('time lr:', (len(self.losses)))
class step_decay_LossHistory(keras.callbacks.Callback):
		def on_train_begin(self, logs={}):
			 self.losses = []
			 self.lr = []
 
		def on_epoch_end(self, batch, logs={}):
				self.losses.append(logs.get('loss'))
				self.lr.append(step_decay(len(self.losses)))
				print('step decay lr:', step_decay(len(self.losses)))
class exp_decay_LossHistory(keras.callbacks.Callback):
		def on_train_begin(self, logs={}):
				self.losses = []
				self.lr = []
				
		def on_epoch_end(self, batch, logs={}):
				self.losses.append(logs.get('loss'))
				self.lr.append(exp_decay(len(self.losses)))
				print('exp decay lr:', exp_decay(len(self.losses)))

#-------------------------------
def CreateModel(feature_columns, 
		n_class,
		optimizer,
		selected_model,
		dropout,
		regul,
		activation_func='relu',
		momentum=0.99
		):
		
		## CREATE THE DNN
		#Create a feature layer : input the feature_columns to our Keras model
		feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
		
		
		#Create the DNN Sequential layers
		model= SetModel(n_class,activation_func,feature_layer,model=selected_model,dropout=dropout,regul=regul,momentum=momentum)


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
		elif model=="model_medium":return model_medium
		elif model=="model_large":return model_large
		elif model=="model_vlarge":return model_vlarge
		
		elif model=="model_small_drop":return model_small_drop
		elif model=="model_medium_drop":return model_medium_drop
		elif model=="model_mlarge_drop":return model_mlarge_drop
		elif model=="model_large_drop":return model_large_drop
		
		elif model=="model_vsmall_L2Regul":return model_vsmall_L2Regul
		elif model=="model_small_L2Regul":return model_small_L2Regul
		elif model=="model_large_L2Regul":return model_large_L2Regul
		elif model=="model_vlarge_L2Regul":return model_vlarge_L2Regul
		
		elif model=="model_medium_L1Regul":return model_medium_L1Regul
		elif model=="model_large_L1Regul":return model_large_L1Regul
		
		elif model=="model_medium_BatchNormalization":return model_medium_BatchNormalization
		elif model=="model_large_BatchNormalization":return model_large_BatchNormalization
		elif model=="model_large_layer1_BatchNormalization":return model_large_layer1_BatchNormalization
		elif model=="model_large_Droupout_BatchNormalization": return model_large_Droupout_BatchNormalization
		
		elif model=="model_large_L2Regul_BatchNormalization":return model_large_L2Regul_BatchNormalization
		elif model=="model_large_L2Regul_drop_BatchNormalization" : return model_large_L2Regul_drop_BatchNormalization
		elif model=="model_vlarge_L2Regul_BatchNormalization":return model_vlarge_L2Regul_BatchNormalization
		
		elif model=="model_large_L1L2Regul_BatchNormalization": return model_large_L1L2Regul_BatchNormalization

def predictDLModel(
		feature_columns, 
		label_name, 
		valid_df, 
		version,
		verbose=2
		):
		TruerainedModel, training_history, level1_DLpred_df, NN_clf = trainDLModel(
																																feature_columns = feature_columns,
																																label_name      = label_name,
																																train_df        = [], 
																																test_df         = [], 
																																valid_df        = valid_df,
																																version         = version,
																																train           = "predict"
																														)
		return level1_DLpred_df
		
def evaluateDLModel(
		feature_columns, 
		label_name, 
		train_df, 
		test_df, 
		valid_df, 
		version,
		verbose=2
		):
		TruerainedModel, training_history, level1_DLpred_df, NN_clf = trainDLModel(
																																feature_columns = feature_columns,
																																label_name      = label_name,
																																train_df        = train_df, 
																																test_df         = test_df, 
																																valid_df        = valid_df,
																																version         = version,
																																train           = "evaluate"
																														)
		return level1_DLpred_df, NN_clf

def trainDLModel(
		feature_columns, 
		label_name, 
		train_df, 
		test_df, 
		valid_df, 
		version,
		train="train",
		verbose=2
		):

		
		if(train!="predict"):
				print("")
				print("======> Deep Neural Network:")

		dateTimeObj = datetime.now()
		timestampStr = dateTimeObj.strftime("%Y-%m-%d")

		# Include the epoch in the file name (uses `str.format`)
		checkpoint_path = "trainingModels/model-cp-"+label_name+"-"+version+".ckpt"
		checkpoint_dir  = os.path.dirname(checkpoint_path)

		#----
		reset_random_seeds()

		#Get DNN HyperParameters
		activation_func, n_class, batch_size, momentum, dropout, selected_model, regul, decay, learning_rate, epochs = DLHyperParameters(label_name)
		
		## Set Optimizer
		if decay=="time":
				#Time base decay
				decay_rate = learning_rate / epochs
				#momentum = 0.99
				#opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
				#opt = opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
				#opt = tf.keras.optimizers.SGD(learning_rate = learning_rate, decay = decay_rate, momentum = momentum)
				optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum = momentum)
		else:
				#opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
				#opt = tf.keras.optimizers.SGD(learning_rate=learning_rate,nesterov=True)
				#opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
				#opt = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
				optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum = momentum)
				
		
		## Create the DNN model
		model = CreateModel(feature_columns,
												n_class,optimizer = optimizer,
												selected_model = selected_model,
												dropout = dropout,
												regul = regul,
												activation_func = activation_func,
												momentum = momentum)

		modelCP = CreateModel(feature_columns,
												n_class,optimizer = optimizer,
												selected_model = selected_model,
												dropout = dropout,
												regul = regul,
												activation_func = activation_func,
												momentum = momentum)

		valid_ds = df_to_dataset(valid_df, label_name, batch_size=batch_size, shuffle=False)

		#----
		if(train=="train" or train=="evaluate"):
				# Convert dataframes to tf.datasets
				#----
				train_ds = df_to_dataset(train_df, label_name, batch_size=batch_size)
				test_ds  = df_to_dataset(test_df , label_name, batch_size=batch_size, shuffle=False)
				

				#upsampling training dataset
				#if (label_name=="totgoal2"): train_df = smotenc_upsample(train_df,label_name)

				classes_zero = train_df[train_df[label_name] == 0]
				classes_one  = train_df[train_df[label_name] == 1]
				classes_two  = train_df[train_df[label_name] == 2]

				# Print Class sizes
				print(" -> DNN Train df Class distribution:")
				print(f'    o Class 0: {len(classes_zero)}')
				print(f'    o Class 1: {len(classes_one)}')
				if(len(classes_two)>0) : print(f'    o Class 2: {len(classes_two)}',"\n")
				print("\n")

				# Convert parts into NumPy arrays for weight computation
				zero_numpy = classes_zero[label_name].to_numpy()
				one_numpy  = classes_one[label_name].to_numpy()
				two_numpy  = classes_two[label_name].to_numpy()
				
				if(len(classes_two)>0):
						all_together = np.concatenate((zero_numpy, one_numpy, two_numpy))
				else:
						all_together = np.concatenate((zero_numpy, one_numpy))
						
				unique_classes = np.unique(all_together)

				# Compute weights
				class_weights = class_weight.compute_class_weight('balanced', unique_classes, all_together)
				
				#----
				print(" -> DNN hyperparameters : ")
				print(f"    o label   : ",label_name)
				print(f"    o optimizer   : ",optimizer)
				print(f"    o model       : ",selected_model)

				if("L1" in selected_model)    : print(f"    o Ruglarization : L1")
				if("L2" in selected_model)    : print(f"    o Ruglarization    : L2")
				if("Regul" in selected_model) : print(f"    o Ruglarization rate: ",regul)

				print(f'    o  batch_size  : ',batch_size)
				print(f'    o  dropout     : ',dropout)
				print(f'    o momentum    : ',momentum,"\n")

				if(train=="evaluate"): #load fitted model

						#--------- Predict Model ----------
						### Load the previously saved weights
						print("Load saved DN model ...")
						model.load_weights(checkpoint_path).expect_partial()
						history=""

				else: #Fit model

						# Create a callback that saves the model's weights every 5 epochs
						cp_callback = tf.keras.callbacks.ModelCheckpoint(
								filepath = checkpoint_path, 
								verbose = verbose, 
								save_weights_only = True,
								period = 5)
						
						custom_early_stopping = EarlyStopping(
								#monitor='val_accuracy'
								monitor = 'val_loss'
								, mode = 'min'    
								#, mode = 'max'
								, patience = 5
								, min_delta = 0.002
						)
						
						#-- Decays 
						callbacks_list = [cp_callback, custom_early_stopping]
						if decay=="time":
								loss_history = LossHistory()
								lrate = learning_rate
								callbacks_list = [loss_history,cp_callback]
						elif decay=="step":
								loss_history = step_decay_LossHistory()
								#Callback Step decay learning rate
								lrate = LearningRateScheduler(step_decay)
								callbacks_list = [loss_history, lrate, cp_callback]
						elif decay=="exponential":
								loss_history = exp_decay_LossHistory()
								#Callback Step decay learning rate
								lrate = LearningRateScheduler(exp_decay)
								callbacks_list = [loss_history, lrate, cp_callback]

						#--------- FIT Model ----------
						if (label_name=="Ov05"
								or label_name=="Ov35"
								or label_name=="FTR"):
								history = model.fit( 
										train_ds,
										validation_data=test_ds,
										verbose=verbose,
										epochs=epochs,
										callbacks=callbacks_list,
										class_weight=class_weights
								)
						else:
								history = model.fit( 
										train_ds,
										validation_data=test_ds,
										verbose=verbose,
										epochs=epochs,
										callbacks=callbacks_list
								)
						
				#--------- Evaluate Model ----------
				print("-> Training Evaluation:  ")
				train_acc = model.evaluate(train_ds, verbose=2)

				print("-> Test Evaluation:  ")
				test_acc  = model.evaluate(test_ds , verbose=2)

				if(train=="train"):
						print("-> Test history : Accuracy: ", history.history["val_accuracy"][-1], 
							" ||   Loss: ", history.history["val_loss"][-1])

		else:
				#--------- Predict Model ----------
				### Load the previously saved weights
				model.load_weights(checkpoint_path).expect_partial()
				history=""

		#--------- Predict data ----------
		preds = model.predict(valid_ds)

		if(train=="train" or train=="evaluate"):
				print("-> Validation predictions ...",len(preds) )
				print( classification_report( valid_df[label_name], np.around(model.predict(valid_ds)) ) )

		#--- Predictions dataframes
		if(label_name=="FTR"):
				predictions= pd.DataFrame( data = preds, columns=(['D_DL', 'W_DL', 'L_DL']) ) 
		else:
				predictions= pd.DataFrame( data = preds, columns=([label_name + '_DL']) )

		predictions = predictions.reset_index(drop=True)

		level1_DLpred_df = pd.concat( [ valid_df['fixture_id'], valid_df[label_name], predictions ] , axis=1)

		#----
		NN_clf = KerasClassifier(build_fn=modelCP, epochs=epochs, batch_size= batch_size)

		#----
		return model, history, level1_DLpred_df, NN_clf

def DLHyperParameters(label_name):
		learning_rate   = 0.000004539992976248485
		mt              = "model_large_drop"
		decay           = "exponential"    #"step" #time
		epochs          = 100; 
		regul           = 0.00001
		activation_func = 'relu' #relu #sigmoid

		if label_name=="FTR":
				n_class = 3 ; batch_size = 226 ; momentum = 0.92 ; dropout=0.092 ; mt = "model_large_L2Regul" ; regul=0.0000001
		
		#--
		elif label_name=="Team1Win" :
				n_class = 1 ; batch_size = 226 ; momentum = 0.92 ; dropout=0.089 ; mt = "model_large_L2Regul" ; regul=0.000000001 
				#epochs=58
		elif label_name=="NotTeam1Win" :
				n_class = 1 ; batch_size = 226 ; momentum = 0.85 ; dropout=0.089 ; mt = "model_large_L2Regul" ; regul=0.00000003 
				
				
		elif label_name=="Team2Win":
				n_class = 1 ; batch_size = 192 ; momentum = 0.92 ; dropout=0.2 ; mt = "model_large_L2Regul" ; regul=0.000000001

		elif label_name=="NotTeam2Win":
				n_class = 1 ; batch_size = 226 ; momentum = 0.85 ; dropout=0.097 ; mt = "model_large_L2Regul" ; regul=0.000002
				
				
		elif label_name=="TeamNWin":
				n_class = 1 ; batch_size = 192 ; momentum = 0.85 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.000000001 
				#learning_rate = 0.0000001 ; decay="time";epochs        = 250; 
				
		#--    
		elif label_name=="Ov05":
				n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.05 ; mt = "model_large_L2Regul" ; regul=0.00002 ; 
				mt = "model_large_L2Regul_drop_BatchNormalization" ; 
				learning_rate = 0.000001 ; decay="time" ; epochs = 150; 
				
		elif label_name=="Ov15":
				n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.000008 ; 
				mt = "model_large_L2Regul_drop_BatchNormalization" ; 
				learning_rate = 0.000002 ; decay="time" ; #epochs = 250; 
				
		elif label_name=="Ov25":
				n_class = 1 ; batch_size = 226 ; momentum = 0.87 ; dropout=0.15 ; mt = "model_large_L2Regul" ; regul=0.0000002 

		elif label_name=="Ov35":
				n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.3 ; mt = "model_large_L2Regul" ; regul=0.000008 ; 
				mt = "model_large_L2Regul_drop_BatchNormalization" ; 
				learning_rate = 0.000002 ; decay="time" ; #epochs = 250; 
				#feature_columns,train_df,valid_df= formatSplitData(dataframeInput,label_name,0.8) ;
				
		elif label_name=="Ov45":
				n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.00008 ; 
				mt = "model_large_L2Regul_drop_BatchNormalization" ; 
				learning_rate = 0.000002 ; decay="time" ; #epochs = 250; 
				#feature_columns,train_df,valid_df= formatSplitData(dataframeInput,label_name,0.8);
				
		elif label_name=="LDEM":
				n_class = 1 ; batch_size = 226 ; momentum = 0.80 ; dropout=0.04 ; mt = "model_large_drop" ; regul=0.00000008
				#feature_columns,train_df,valid_df= formatSplitData(dataframeInput,label_name,0.75);
		#--    
		elif label_name=="totgoal0":
				n_class = 1 ; batch_size = 226 ; momentum = 0.80 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.000008
				mt = "model_large_L2Regul_drop_BatchNormalization" ; 
				#learning_rate = 0.00003 ; decay="time" #; epochs = 200; 

		elif label_name=="totgoal1":
				n_class = 1 ; batch_size = 226 ; momentum = 0.80 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.000008
				mt = "model_large_L2Regul_drop_BatchNormalization" ; 
				learning_rate = 0.00004 ; decay="time" #; epochs = 200; 

		elif label_name=="totgoal2":
				n_class = 1 ; batch_size = 226 ; momentum = 0.99 ; dropout=0.5 ; mt = "model_large_L2Regul" ; regul=0.00001 ; 
				mt = "model_large_L2Regul_drop_BatchNormalization" ; 
				learning_rate = 0.000002 ; decay="time" ; #epochs = 250; 
				#feature_columns,train_df,valid_df= formatSplitData(dataframeInput,label_name,0.8);

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
				#feature_columns,train_df,valid_df= formatSplitData(dataframeInput,label_name,0.82);
		#----
		return activation_func, n_class, batch_size, momentum, dropout, mt, regul, decay, learning_rate, epochs

#-------- MACHINE LEARNING ------

def MLPredict(label_name, df,version,model_type='RFC-stack'):
		
		filename,encoderfilename= FileNames(label_name,version,model_type=model_type)
		#-
		load_model = pickle.load(open(filename, 'rb'))
		
		reset_random_seeds()
		prediction  = load_model.predict(df)

		labl_enc = LabelEncoder()
		labl_enc.classes_ = np.load(encoderfilename)

		prediction    = labl_enc.inverse_transform(prediction)
		prediction    = pd.DataFrame(data = prediction, columns=([label_name]))
		
		return prediction

def MLHyperParameters(label_name,clf):

		if (label_name=="FTR"):
				min_samples_leaf=3 ; max_features=16 ; max_depth = 31 ;
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }

		elif (label_name=="LDEM"):
				min_samples_leaf=2 ; max_features=15 ; max_depth = 13 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
				
		elif label_name=="Team1Win":
				min_samples_leaf=3 ; max_features=23 ; max_depth = 15 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
		elif label_name=="NotTeam1Win":
				min_samples_leaf=1 ; max_features=33 ; max_depth = 10 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
				
		elif label_name=="Team2Win":
				min_samples_leaf=1 ; max_features=25 ; max_depth = 8 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
		elif label_name=="NotTeam2Win":
				min_samples_leaf=1 ; max_features=19 ; max_depth = 34 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
				
		elif label_name=="TeamNWin":
				min_samples_leaf=1 ; max_features=10 ; max_depth = 26 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
		elif label_name=="NotTeamNWin":
				min_samples_leaf=3 ; max_features=11 ; max_depth = 14 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
				
		elif label_name=="Ov05":
				min_samples_leaf=1 ; max_features=6 ; max_depth = 35 ; 
				param_grid = {'max_depth': [13], 'max_features': [36], 'min_samples_leaf': [2]}
		elif label_name=="Ov15":
				min_samples_leaf=2 ; max_features=23 ; max_depth = 11 ;
				param_grid = {'max_depth': [14], 'max_features': [35], 'min_samples_leaf': [2]}
				
		elif label_name=="Ov25":
				min_samples_leaf=3 ; max_features=8  ; max_depth = 32 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
		elif label_name=="Ov35":
				min_samples_leaf=2 ; max_features=4  ; max_depth = 15 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
		elif label_name=="Ov45":
				min_samples_leaf=1 ; max_features=3  ; max_depth = 18 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
	 
		elif label_name=="totgoal0":
				min_samples_leaf=1 ; max_features=16 ; max_depth = 34 ;
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
		elif label_name=="totgoal1":
				min_samples_leaf=1 ; max_features=24 ; max_depth = 21 ;
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
		elif label_name=="totgoal2":
				min_samples_leaf=2 ; max_features=32 ; max_depth = 32 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
		elif label_name=="totgoal3":
				min_samples_leaf=9 ; max_features=23 ; max_depth = 28 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
		elif label_name=="totgoal4":
				min_samples_leaf=3 ; max_features=14 ; max_depth = 17 ; 
				#param_grid = {  'min_samples_leaf' : [3], 'max_features': [16], 'max_depth': [31] }
		elif label_name=="totgoal5":
				min_samples_leaf=1 ; max_features=35 ; max_depth = 10 ; 
				#param_grid = {  'min_samples_leaf' : [1], 'max_features': [35], 'max_depth': [10] }

		if (clf=="RandomForestClassifier" ): return param_grid, min_samples_leaf, max_features, max_depth

def evaluateMLModel(
		label_name, 
		train_df, 
		test_df, 
		valid_df, 
		version,
		model_type="RFC"
		):
		level1_MLpred_df, RF_clf = trainMLModel( label_name, train_df,  test_df,  valid_df,  version, "evaluate",model_type=model_type,
		predictProba=True )
		return level1_MLpred_df, RF_clf

def predictMLModel(
		label_name, 
		valid_df, 
		version,
		model_type="RFC",
		predictProba=True
		):

		level1_MLpred_df, RF_clf = trainMLModel( label_name,  [],  [],  valid_df,  version, "predict",model_type=model_type,
		predictProba=True )
		return level1_MLpred_df

def trainMLModel(
		label_name, 
		train_df, 
		test_df, 
		valid_df, 
		version,
		train="train",
		verbose=2,
		model_type="RFC",
		predictProba=True
		):

		if(train!="predict"):
				print("")
				print("======> Machine Learning START .... ")

		version = "-" + version

		#File name definition
		directory = 'trainingModels/model-cp-' + model_type + '-'
		filename        = directory + label_name  + version + '.sav'
		encoderfilename = directory +'encoder-'  + label_name + version + '.npy'

		#--
		x_valid = valid_df.drop([label_name],axis=1)
		y_valid = valid_df[label_name]


		#ValidNbfeatures=len(x_valid. columns)
		#if(train!="predict"):
		#    print("Nb of features for validation set:",ValidNbfeatures)

		#display(x_valid.head())
		
		#--
		if(train=="train" or train=="evaluate"):
				
				print("")
				print("==> Start Training - ", label_name) 
				reset_random_seeds()

				#Unbalanced - resampling
				#if(label_name=="Team2Win"):
				#    train_df = resample("over", train_df, label_name) #under,over

				#--
				x_train = train_df.drop([label_name],axis=1)
				y_train = train_df[label_name]

				x_test = test_df.drop([label_name],axis=1)
				y_test = test_df[label_name]


				if(train=="evaluate"):
						print("loading saved ML model... ", filename) 

						model = pickle.load(open(filename, 'rb'))

				else:

						#Cross-Validation parameters
						#--
						class_weights = dict( 
									zip(np.unique(y_train)
								, class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)) 
								)

						#----
						Nbfeatures=len(x_train. columns)
						print("Nbfeatures:",Nbfeatures)

						#-- model - Classifier & param_grid 
						if (model_type=="RFC"):
								level0_clf = RandomForestClassifier(random_state = 2 , n_estimators = 100, class_weight='balanced')
								param_grid = {  
										#'max_features': range(1,Nbfeatures,1) ,
										'max_depth' : range(6, 14, 1),
										'min_samples_leaf' : [3,4,5] #range(1, 4, 1)
								}


						elif (model_type=="XGB"):
								level0_clf = XGBClassifier(random_state = random.seed(1234) , n_estimators = 100)
								param_grid = { # XGBoost
									 #'max_depth': [15,20,25]
									 'reg_alpha': [1.1, 1.2, 1.3]
									 ,'reg_lambda': [1.1, 1.2, 1.3]
									 ,'subsample': [0.7, 0.8, 0.9]
								}

						NbFolds = 5
						if(label_name=="Team2Win"):
								NbFolds = 15

						kfolds = StratifiedKFold(NbFolds)
						refit = 'recall_macro'

						print("GridSearchCV refit = ", refit)


						CV_rfc = GridSearchCV( estimator   = level0_clf
																	, n_jobs     = -1
																	, param_grid = param_grid
																	, cv         = kfolds.split(x_train, y_train)
																	, scoring    = ['accuracy','precision_macro','recall_macro', 'f1_macro']
																	, verbose    = 10
																	, refit      = refit)
						CV_rfc.fit(x_train, y_train)

						print ("best score: ", CV_rfc.best_score_ )
						print ("best params: ", CV_rfc.best_params_)
						
						#Create the model from best_estimators
						model = CV_rfc.best_estimator_
						model.fit(x_train, y_train)

						#Save the model
						print("")
						print("==> Save the model : ",filename)
						pickle.dump(model, open(filename, 'wb'))

				#--
				pred_train  = model.predict(x_train)
				pred_test   = model.predict(x_test)
				print("")
				print("     ... Train - accuracy_score = %1.2f" % accuracy_score(y_train,pred_train))
				print("     ... Test  - accuracy_score = %1.5f" % accuracy_score(y_test,pred_test))
				print("     ... Test  - classification:\n",classification_report(y_test, pred_test))

		elif(train=="predict"):
				#print("loading saved ML model... ", filename) 
				model = pickle.load(open(filename, 'rb'))
				x_valid = x_valid.drop(["event_timestamp"],axis=1)
				
		############## Predict Full dataset Accuracy

		tmpprediction  = model.predict(x_valid)
		tmpprediction_class = pd.DataFrame( data = tmpprediction, columns=([label_name + '_ML_class'])).reset_index(drop=True)

		preds    = model.predict_proba(x_valid)

		if(train=="train" or train=="evaluate"):
				print("Validation predictions ...",len(preds) )
				print("     ... Validation - accuracy_score = %1.3f" % accuracy_score(valid_df[label_name], tmpprediction))
				print("     ... Validation  - classification:\n",classification_report(valid_df[label_name], tmpprediction))

		#--- Predictions dataframes
		if(label_name=="FTR"):
				predictions= pd.DataFrame( data = preds, columns=(['D_ML', 'W_ML', 'L_ML']) ) 
		else:
				predictions= pd.DataFrame( data = preds, columns=([label_name + '_ML','not_'+label_name + '_ML']) ).drop(['not_'+label_name + '_ML'],axis=1)

		predictions = predictions.reset_index(drop=True)

		#---
		level1_MLpred_df = pd.concat( [ valid_df['fixture_id'],valid_df[label_name], predictions ] , axis=1)

		if(label_name=="TeamNWin"):
				level1_MLpred_df = pd.concat( [ level1_MLpred_df ,tmpprediction_class ] , axis=1)

		if(predictProba):
				return level1_MLpred_df , model
		else:
				#printt("test")
				tmpprediction = tmpprediction.reset_index(drop=True)
				return tmpprediction_class , model
	
#-------- STACKING ------   

def CVtrain(x_train, x_test, y_train, y_test,label_name,version, train=True,stack=False):
		
		if(stack):
				filename,encoderfilename = FileNames(label_name,version,model_type='RFC-stack')
		else:
				filename,encoderfilename = FileNames(label_name,version,model_type='RFC')

				
		if(train):
				
				Nbfeatures=len(x_train. columns)
				print("Nbfeatures:",Nbfeatures)
				
				print("")
				print("==> Start CV Training - ", label_name) 


				class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced',
																								 np.unique(y_train),
																								 y_train))) 
				
				#--
				#Random forest Meta Learner

				level1_clf = RandomForestClassifier(
						random_state     = random.seed(1234)
						#,class_weight    = class_weights
						,class_weight='balanced'
				)

				param_grid = {  
						'n_estimators' : [100,150,200,250] ,
						#'max_features': range(1,Nbfeatures,1) ,
						'max_depth' : range(8, 17, 1),
						'min_samples_leaf' : range(1, 3, 1)
				}
				
				#--
						#clf = GradientBoostingClassifier(random_state = random.seed(1234) , n_estimators = 100)

						#--
						#clf = ExtraTreesClassifier(random_state = random.seed(1234) , n_estimators = 100)
						#param_grid={
						#    #'max_features': range(50,401,50),
						#    'min_samples_leaf': range(20,50,5),
						#    #'min_samples_split': range(15,36,5),
						#}

						#--
						# level1_clf = XGBClassifier(random_state = random.seed(1234) , n_estimators = 100)
						# param_grid = { # XGBoost
						#    'max_depth': [15,20,25],
						#    'reg_alpha': [1.1, 1.2, 1.3],
						#    'reg_lambda': [1.1, 1.2, 1.3],
						#    'subsample': [0.7, 0.8, 0.9]
						# }

				#--
				
				reset_random_seeds();
				NbFolds = 5
				if(label_name=="Team2Win"):
						NbFolds=10
				kfolds = StratifiedKFold(NbFolds)

				refit = 'f1_macro'

				print("GridSearchCV refit = ", refit)
				CV_clf = GridSearchCV( estimator   = level1_clf
															, n_jobs     = -1
															, param_grid = param_grid
															, cv         = kfolds.split(x_train, y_train)
															, scoring    = ['accuracy','precision_macro','recall_macro', 'f1_macro']
															, verbose    = 10
															, refit      = refit)

				CV_clf.fit(x_train, y_train)

				print ("best score : ",CV_clf.best_score_ )
				#print ("best params : ",CV_clf.best_params_)
				print ("best estimator : ",CV_clf.best_estimator_)

				model = CV_clf.best_estimator_
				model.fit(x_train, y_train)
				#Save the model
				pickle.dump(model, open(filename, 'wb'))

		else:

				#Load model
				print("model ", filename, " loaded")
				model = pickle.load(open(filename, 'rb'))

		return model

def trainStackModel(data, label_name, train, version):
		
		filename,encoderfilename= FileNames(label_name, version, model_type='RFC-stack')
		splitratio=0.5

		if(train=="train" or train=="evaluate"):
				print("Split Data : training/validation")
				TrendX_all, TrendY_all, x_train, x_test, y_train, y_test, lab_enc = splitData(data
																																										,label_name
																																										,splitratio)
				if(train=="evaluate"): 
						print("load saved stacked model...")
						clf = pickle.load(open(filename, 'rb'))
				else:
						clf = CVtrain(x_train, x_test, y_train, y_test ,label_name,version, train=True,stack=True)
						#Save the model
						print("")
						print("==> Save the stacked model : ",filename)
						pickle.dump(clf, open(filename, 'wb'))


				pred_train  = clf.predict(x_train)
				accuracy_score_train = accuracy_score(y_train,pred_train)

				pred_test = clf.predict(x_test)
				accuracy_score_test = accuracy_score(y_test,pred_test)

				print("     ... Train - accuracy_score = %1.3f" % accuracy_score_train)
				print("     ... Test  - accuracy_score = %1.3f" % accuracy_score_test)
				print("     ... Test  - classification:\n",classification_report(y_test, pred_test))

				return clf 

		elif(train=="predict"):

				#Load model
				#print("load saved stacked model...")
				clf = pickle.load(open(filename, 'rb'))
				x = data.drop([label_name],1)
				if("event_timestamp" in x) : x = x.drop(["event_timestamp"],1)

				return clf.predict(x)
				
def evaluateStackModel(data,label_name,version): 
		trainStackModel(data, label_name, "evaluate",version)

def predictStackModel(data,label_name,version): 
		preds = trainStackModel(data, label_name, "predict",version)
		predictions= pd.DataFrame( data = preds, columns=([label_name + '_SK']) ).reset_index(drop=True)

		return predictions


#####################################################

def validate():
		version="v5"
		label_name = "Team1Win" 
		#Import Predict data
		df = getDataToValidate()

		dataframe = formatData(df,label_name)
		#display(dataframe.head())

		#Get Full Feature Columns
		feature_columns   = featureColumns(reload=True)

		level0_train_df, level0_test_df, valid_df = stackSplitData(dataframe,label_name)
		
		#display(valid_df.shape)
				

		level1_DLpred_df, NN_clf  = evaluateDLModel(
																								feature_columns = feature_columns,
																								label_name      = label_name,
																								train_df        = level0_train_df, 
																								test_df         = level0_test_df, 
																								valid_df        = valid_df,
																								version         = version
																						)

		level1_MLpred_df, RF_clf    = evaluateMLModel(
																		label_name = label_name, 
																		train_df   = level0_train_df, 
																		test_df    = level0_test_df, 
																		valid_df   = valid_df, 
																		version    = version,
																		model_type      = "RFC"
																)

		data = pd.concat(
		[
					level1_DLpred_df[label_name+"_DL"]
				, level1_MLpred_df[label_name+"_ML"]
				, valid_df
		]
		, axis=1)
		#LEVEL 1
		evaluateStackModel(data,label_name, version)

def predict(date=datetime.now().strftime("%Y-%m-%d"),offset=""):
		version="v5"
		predictions=[]
		#Update Player Ratings
		#getPlayersRatings(True,date,offset)


		#Import Predict data
		df = getDataToPredict(offset,date)
		#df = getDataToValidate()

		if len(df)>0:

				#Get Full Feature Columns
				feature_columns   = featureColumns(reload=True)

				#--
				# Iterate for each labels
				#label_name = "Team1Win" 
				label_names = ["Team1Win","Team2Win","TeamNWin"]
				#label_names = ["Team2Win"]
				for label_name in label_names:

						print("  -> Predicting label :",label_name)
						dataframe = formatData(df,label_name,False)
						valid_df = dataframe

						#df=controlPlayers(df)

						#LEVEL 0
						level1_DLpred_df = predictDLModel(  feature_columns,  label_name,  valid_df, version )
						level1_MLpred_df = predictMLModel(  label_name,   valid_df,   version , model_type = "RFC", predictProba=False)

						data = pd.concat(
						[
									level1_DLpred_df[label_name+"_DL"]
								, level1_MLpred_df[label_name+"_ML"]
								, valid_df
						]
						, axis=1)

						#LEVEL 1
						if (label_name == "Team1Win" ):
								predic = predictStackModel(data,label_name,version)
								#predict1 = predictStackModel(data,label_name,version)
						if (label_name == "Team2Win" ):

								predic = pd.concat([predic, predictStackModel(data,label_name,version)],axis=1)
								#predict2 = predictStackModel(data,label_name,version)
						if (label_name == "TeamNWin" ):
								predic = pd.concat([predic, level1_MLpred_df[label_name + '_ML_class']],axis=1)
								#predict3 = level1_MLpred_df[label_name + '_ML_class']

				#predic = pd.concat([predict1,predict2,predict3],axis=1)
				#predic = predict2

				dataMatchInfo = pd.merge(df, valid_df, on=['fixture_id'], how='inner')
				dataMatchRes = dataMatchInfo[["goalsHomeTeam" , "goalsAwayTeam"]]
				dataMatchInfo = dataMatchInfo[['fixture_id' , "country_name_x", "league_name_x" , "homeTeam_x" , "awayTeam_x"]]

				#display(dataMatchInfo.head())
				#dataMatchRes["totals"] = dataMatchRes['goalsHomeTeam']+dataMatchRes['goalsAwayTeam']


				predictions = pd.concat(
											[  dataMatchInfo
											 , predic
											 #, dfErrors
											 , dataMatchRes
											]
											, axis=1)
		return predictions

#####################################################
###########      DATAFRAME RULES
#####################################################

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

def ratingRanges_v5(x,col):
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

#####################################################
###########       DATA CLEANING    
#####################################################

def df_to_dataset(dataframe, label_name, shuffle=True, batch_size=32):
		dataframe = dataframe.copy()
		
		labels = dataframe.pop(label_name)

		ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))

		if shuffle: ds = ds.shuffle(buffer_size=len(dataframe))

		ds = ds.batch(batch_size)
		return ds

def buildNewNewFeaturesInColumn(dataframeInput,fixturePlayerData,PlayersIDcols,newStats ):
    #Fore each fixture add the columns to the dataset

    for index, row in dataframeInput.iterrows():

        fixture_id = row["fixture_id"]
        league_id  = row["league_id"]

        print("=> row: ",index, "(",100*index/len(dataframeInput) , "%) fixture_id:",row["fixture_id"],"  league_id:",row["league_id"])

        fixturePlayerData = dataframePlayerInput[
              (dataframePlayerInput["Nextfixture_id"] == int(fixture_id))
            & (dataframePlayerInput["league_id"]      == int(league_id))
        ].reset_index(drop=True)

        #display(fixturePlayerData)
        for teamstat in newTeamStat:
            #print(teamstat)
            #print(fixturePlayerData[teamstat])
            if(len(fixturePlayerData[teamstat]>0)):
                #print(fixturePlayerData.at[0,teamstat])

                dataframeInput.at[index,teamstat] = fixturePlayerData.at[0,teamstat]
                #print(teamstat, " ", dataframeInput.at[index,teamstat])

        #for each players in fixture
        for col in PlayersIDcols:

            if(isNotNaN(row[col])):

                player_id  = row[col].replace("ID","")


                print(" - Player Id : ",player_id)

                playerdata = fixturePlayerData[ 
                    ( fixturePlayerData["player_id"] == int(player_id) )
                ].reset_index(drop=True)

                #display(playerdata)

                if(len(playerdata)>0):

                    #Replace Scorecolumns
                    #Add new columns: 1 per add stat

                    for newcol in newStats:
                        print("        Add col : ",newcol, " to :" , col)
                        dataframeInput.at[index,str(col)+"_"+newcol] = playerdata.at[0,newcol]
                else:
                    print("Player (",player_id,") not found in fixture",fixture_id)

    #- Remove Score column
    for col in PlayersIDcols:
        print("drop: "+ str(col)+"Score")
        dataframeInput=dataframeInput.drop([str(col)+"Score"],1)
        
    return dataframeInput

def prepareData(dataframeInput, label_name, HT=False, prints=True
	,PlayersIDcols=[
		'HTPlayerG',
		'HTPlayerD1','HTPlayerD2','HTPlayerD3','HTPlayerD4','HTPlayerD5','HTPlayerD6',
		'HTPlayerM1','HTPlayerM2','HTPlayerM3','HTPlayerM4','HTPlayerM5','HTPlayerM6','HTPlayerM7',
		'HTPlayerF1','HTPlayerF2','HTPlayerF3','HTPlayerF4',

		'HTPlayerG',
		'HTPlayerD1','HTPlayerD2','HTPlayerD3','HTPlayerD4','HTPlayerD5','HTPlayerD6',
		'HTPlayerM1','HTPlayerM2','HTPlayerM3','HTPlayerM4','HTPlayerM5','HTPlayerM6','HTPlayerM7',
		'HTPlayerF1','HTPlayerF2','HTPlayerF3','HTPlayerF4'
	]
	, PlayersRatingcols=[
		'HTPlayerGScore',
		'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
		'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
		'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
		'ATPlayerGScore',
		'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
		'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
		'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score'
		]):
		
		dataframe=dataframeInput.copy()
		
		#Remove unnecessary columns
		dataframe=dataframe.drop([
				"homeUpdated",
				"awayUpdated",
				"statusShort",
				"status"
		], axis=1)
		#display(dataframe.head())
		#Format conversions
		dataframe["event_date"]   = pd.to_datetime(dataframe["event_date"], format='%Y-%m-%d')
		dataframe["LDEM_YES"]     = pd.to_numeric(dataframe["LDEM_YES"], errors='coerce')
		dataframe["Over2_5goals"] = pd.to_numeric(dataframe["Over2_5goals"], errors='coerce')
		dataframe["W1"] 					= pd.to_numeric(dataframe["W1"], errors='coerce')
		dataframe["W2"] 					= pd.to_numeric(dataframe["W2"], errors='coerce')
		dataframe["WN"] 					= pd.to_numeric(dataframe["WN"], errors='coerce')
		dataframe["LDEM_YES"] 		= pd.to_numeric(dataframe["LDEM_YES"], errors='coerce')
		dataframe["Over0_5goals"] = pd.to_numeric(dataframe["Over0_5goals"], errors='coerce')
		dataframe["Over1_5goals"] = pd.to_numeric(dataframe["Over1_5goals"], errors='coerce')
		dataframe["Over2_5goals"] = pd.to_numeric(dataframe["Over2_5goals"], errors='coerce')
		dataframe["Over3_5goals"] = pd.to_numeric(dataframe["Over3_5goals"], errors='coerce')
		dataframe["Over4_5goals"] = pd.to_numeric(dataframe["Over4_5goals"], errors='coerce')
		

		dataframe["round"]        = dataframe["round"].fillna(0.0).astype(int)
		dataframe["HTF"]          = dataframe["HTF"].fillna(0.0).astype(int)
		dataframe["ATF"]          = dataframe["ATF"].fillna(0.0).astype(int)
		
		#Add New Data to the dataset
		dataframe["totgoal"]  		= dataframe['goalsHomeTeam']+dataframe['goalsAwayTeam']
		dataframe["goalDiff"] 		= dataframe['goalsHomeTeam']-dataframe['goalsAwayTeam']
		dataframe["FTHG"]     		= dataframe['goalsHomeTeam']
		dataframe["FTAG"]     		= dataframe['goalsAwayTeam']
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

		#---
		if label_name =="Team1N":    
				dataframe["Team1N"] = np.where( dataframe["FTR"]==1 or dataframe["FTR"]==0, 1, 0)

		if label_name =="Team2N":    
				dataframe["Team1N"] = np.where( dataframe["FTR"]==2 or dataframe["FTR"]==0, 1, 0)
		#---

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

		#---
		##Column Groups
		BetCols = ['BW1', 'BWN', 'BW2',
				'BOver1_5goals', 'BUnder1_5goals','BOver2_5goals', 'BUnder2_5goals','BOver3_5goals', 'BUnder3_5goals',
				'BLDEM_YES', 'BLDEM_NO']
		
		
		dataframe[PlayersIDcols] = dataframe[PlayersIDcols].replace({'0':"IDnull", 0:"IDnull","0":"IDnull"})
		dataframe[PlayersIDcols] = dataframe[PlayersIDcols].replace({np.nan:"IDnull"})
		
		
		TeamRatingcols=['HTPlayerDScore','HTPlayerMScore','HTPlayerFScore','ATPlayerDScore','ATPlayerMScore','ATPlayerFScore']
		

		
		#Teams Scores new columns

		if PlayersRatingcols[0] in dataframe.columns: dataframe[PlayersRatingcols] = dataframe[PlayersRatingcols].replace({0.0:np.nan})
		
		dataframe['HTPlayerDScore'] = dataframe[['HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score']].mean(axis=1)
		dataframe['HTPlayerMScore'] = dataframe[['HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score']].mean(axis=1)
		dataframe['HTPlayerFScore'] = dataframe[['HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score']].mean(axis=1)

		dataframe['ATPlayerDScore'] = dataframe[['ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score']].mean(axis=1)
		dataframe['ATPlayerMScore'] = dataframe[['ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score']].mean(axis=1)
		dataframe['ATPlayerFScore'] = dataframe[['ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score']].mean(axis=1)

		
		if PlayersRatingcols[0] in dataframe.columns: dataframe[PlayersRatingcols] = dataframe[PlayersRatingcols].replace({np.nan:0.0})

		# Add League avg success on first half
		#gp=dataframe.groupby(['league_id'])
		
		#display(gp['league_id','homeWin_halftime','awayWin_halftime'].head())
		#dataframe["homeWin_halftime"] = gp['homeWin_halftime'].agg(np.mean)
		#dataframe["awayWin_halftime"] = gp['awayWin_halftime'].agg(np.mean)
		
		## Add League avg success on first half
		if HT==False:
				## download halftimeMean 
				path="https://web-concepts.fr/soccer/win-pronos/pronos-R-data-dev.php?action=halftimeMean"
				df_halftime = GetCSVtoDF(path)

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
		t=len(dataframe)

		dataframe = dataframe[dataframe["season"] >2016]
		if(prints): print("remove from season: ",t-len(dataframe))
		t=len(dataframe)
		dataframe = dataframe[dataframe["round"] >=2]
		if(prints): print("remove from rounds: ",t-len(dataframe))
		t=len(dataframe)
		
		dataframe = dataframe[dataframe.BW1 >0]
		if(prints): print("remove from BW1: ",t-len(dataframe))
		t=len(dataframe)
		dataframe = dataframe[dataframe.BW2 >0]
		if(prints): print("remove from BW2: ",t-len(dataframe))
		t=len(dataframe)
		dataframe = dataframe[dataframe.BWN >0]
		if(prints): print("remove from BWN: ",t-len(dataframe))
		t=len(dataframe)
		
		#display(dataframe.head())
		
		dataframe = dataframe[dataframe["W1"].between(0, 1)]
		dataframe = dataframe[dataframe["W2"].between(0, 1)]
		dataframe = dataframe[dataframe["WN"].between(0, 1)]
		dataframe = dataframe[dataframe["LDEM_YES"].between(0, 1)]
		#dataframe = dataframe[dataframe["Over0_5goals"].between(0, 1)]
		dataframe = dataframe[dataframe["Over1_5goals"].between(0, 1)]
		dataframe = dataframe[dataframe["Over2_5goals"].between(0, 1)]
		dataframe = dataframe[dataframe["Over3_5goals"].between(0, 1)]
		dataframe = dataframe[dataframe["Over4_5goals"].between(0, 1)]
		
		
		#exclude missing player data
		dataframe=dataframe[dataframe["HTPlayerG"] !="IDnull"]
		if(prints): print("remove from HTPlayerG: ",t-len(dataframe))
		t=len(dataframe)

		#dataframe=dataframe[dataframe["HTPlayerD1"]!="IDnull"]
		#dataframe=dataframe[dataframe["HTPlayerM1"]!="IDnull"]
		#dataframe=dataframe[dataframe["HTPlayerF1"]!="IDnull"]  
		
		#dataframe=dataframe[dataframe["ATPlayerG"] !="IDnull"]
		#if(prints): print("remove from ATPlayerG: ",t-len(dataframe))
		#t=len(dataframe)
		
		#dataframe=dataframe[dataframe["ATPlayerD1"]!="IDnull"]
		#dataframe=dataframe[dataframe["ATPlayerM1"]!="IDnull"]
		#dataframe=dataframe[dataframe["ATPlayerF1"]!="IDnull"]  
		
		
		t2=len(dataframe)

		if(prints): print("=>Clean revoved data: ",t-len(dataframe))
		##---- Data Normalization
		
		#Probablity new columns
		dataframe["DiffWin12"] = dataframe["W1"]-dataframe["W2"]
		dataframe["DiffWin1N"] = dataframe["W1"]-dataframe["WN"]
		dataframe["DiffWin2N"] = dataframe["W2"]-dataframe["WN"]

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
		
		
		#std_scaler    = StandardScaler()
		#rbst_scaler   = RobustScaler()
		othercol=['season','round','HTP','HTGS','HTGC','HTF','ATP','ATGS','ATGC','ATF','HTGD','ATGD']
		
		ColsToScale= BetCols + PlayersRatingcols  + othercol #+ TeamRatingcols
		
		
		#minmax_scaler = MinMaxScaler()
		#scaled_df = minmax_scaler.fit_transform(dataframe[ColsToScale])
		#dataframe[ColsToScale] = pd.DataFrame(scaled_df, columns=ColsToScale)
		## save the scaler
		#dump(minmax_scaler, open('scaler.pkl', 'wb'))

		#Fill NANs to 0
		dataframe = dataframe.fillna(0)

		##----
		#Country filtering
		#dataframe=dataframe[dataframe["country_name"]!="Denmark"]
		
		
		##----
		#Drop hiddens columns
		dataframe = dataframe.drop(['event_date','totgoal','goalDiff','goalsHomeTeam','goalsAwayTeam',
																'FTHG','FTAG','pred_match_winner' ,'country_code'], axis=1)  
		
		
		if label_name !="FTR":
				dataframe = dataframe.drop(['FTR'] , axis=1)  

		
		dataframe = dataframe.drop(PlayersIDcols,1, errors='ignore')
		
		##----
		#Mélange les données
		dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
		

		return dataframe

def EncodeFeatureCategoricalLabel(dataframe, load=False,PlayersRatingcols=[]):

		encoderfilename = 'trainingModels/featureCategoricalLabelEncoder.npy'
		labl_enc = LabelEncoder()
		features_categorical_gen=[
						'region_name', 'country_name', 'league_name', 'homeTeam', 'awayTeam'
				]
		features_categorical = features_categorical_gen + PlayersRatingcols
		if(load):
				labl_enc.classes_ = np.load(encoderfilename)

		for cat in features_categorical:    
				 dataframe[cat]= labl_enc.fit_transform(dataframe[cat]) 

		if(load==False):
				np.save(encoderfilename, labl_enc.classes_)

		return dataframe

def decodeFeatureCategoricalLabel(dataframe,PlayersRatingcols=[]):

		encoderfilename = 'trainingModels/featureCategoricalLabelEncoder.npy'
		labl_enc = LabelEncoder()
		
		features_categorical_gen=[
						'region_name', 'country_name', 'league_name', 'homeTeam', 'awayTeam'
				]
		features_categorical = features_categorical_gen + PlayersRatingcols
		

		labl_enc.classes_ = np.load(encoderfilename)

		for cat in features_categorical:    
				 dataframe[cat]= labl_enc.inverse_transform(dataframe[cat]) 

		return dataframe

def formatSplitData(dataframeInput, label_name, split_ratio):

		dataframe         = formatData(dataframeInput,label_name)
		feature_columns   = featureColumns(dataframe)

		train_df, test_df = SimpleSplitData(dataframe,split_ratio)

		return feature_columns, train_df, test_df

def formatData(dataframeInput, label_name, prints=True
	,PlayersIDcols=[
		'HTPlayerG',
		'HTPlayerD1','HTPlayerD2','HTPlayerD3','HTPlayerD4','HTPlayerD5','HTPlayerD6',
		'HTPlayerM1','HTPlayerM2','HTPlayerM3','HTPlayerM4','HTPlayerM5','HTPlayerM6','HTPlayerM7',
		'HTPlayerF1','HTPlayerF2','HTPlayerF3','HTPlayerF4',

		'HTPlayerG',
		'HTPlayerD1','HTPlayerD2','HTPlayerD3','HTPlayerD4','HTPlayerD5','HTPlayerD6',
		'HTPlayerM1','HTPlayerM2','HTPlayerM3','HTPlayerM4','HTPlayerM5','HTPlayerM6','HTPlayerM7',
		'HTPlayerF1','HTPlayerF2','HTPlayerF3','HTPlayerF4'
	]
	, PlayersRatingcols=[
		'HTPlayerGScore',
		'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
		'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
		'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
		'ATPlayerGScore',
		'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
		'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
		'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score'
		]):

		if(prints) : 
				print("");
				print("======> Prepare data for label: ",label_name)


		dataframe = prepareData(dataframeInput, label_name, False, prints, PlayersIDcols, PlayersRatingcols)
		dataframe = EncodeFeatureCategoricalLabel(dataframe, False, PlayersRatingcols)
		if(prints) : 
				print("=> Available data shape... ",dataframe.shape)
				print("---------------------------")

		return dataframe

def StarifySplitData(dataframe, label_name, splitratio):

		#Split the dataframe into train, validation
		#----
		TrendX_all = dataframe.drop([label_name],axis=1);
		TrendY_all = dataframe[label_name];

		#----
		X_train, X_test, y_train, y_test = train_test_split(TrendX_all, TrendY_all,  test_size = int(len(dataframe)*splitratio),  random_state = 2, stratify = TrendY_all)
		
		train_df = X_train
		train_df[label_name] = y_train

		test_df = X_test
		test_df[label_name] = y_test

		print(len(train_df), 'train examples')
		print(len(test_df) , 'validation examples')

		return train_df, test_df

def SimpleSplitData(dataframe, split_ratio):

		#Split the dataframe into train, validation
		train_max_row = int(dataframe.shape[0]*split_ratio)

		#----
		train_df = dataframe.iloc[:train_max_row]
		valid_df = dataframe.iloc[train_max_row:]

		#----
		print(len(train_df), 'train examples')
		print(len(valid_df), 'validation examples')

		return train_df, valid_df

def splitData(dataframe, label_name, splitratio, prints=True, encode=True):
		lab_enc = LabelEncoder()
		
		### Set the label : NextTrend
		TrendX_all = dataframe.drop([label_name],1)
		TrendY_all = dataframe[label_name]

		if(encode==True):
				TrendY_all = lab_enc.fit_transform(TrendY_all)

		X_train=[];  X_test=[];  y_train=[];  y_test=[]; 
		
		#### Split data to Training/Test
		testSize = int(len(dataframe)*splitratio)

		try:
				X_train, X_test, y_train, y_test = train_test_split(TrendX_all, TrendY_all, test_size = testSize,  random_state = 2, stratify = TrendY_all)
		except:
				print("Warning: ",label_name," Not Stratified")
				X_train, X_test, y_train, y_test = train_test_split(TrendX_all, TrendY_all, test_size = testSize, random_state = 2)

		#----
		if prints==True:
				print(len(X_train), ' train examples')
				print(len(X_test) , ' test examples' )

		return TrendX_all, TrendY_all, X_train, X_test, y_train, y_test, lab_enc

def featureColumns(reload=True,dataframe=[],
		feature_numeric_player_stats = [
				'HTPlayerGScore',
				'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
				'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
				'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
				'ATPlayerGScore',
				'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
				'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
				'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score']):
		
		if(reload==True):
				dataframe = pd.read_csv("featureColumns-"+version+".csv", sep="\t", encoding = "ISO-8859-1", index_col=False, low_memory=False)
		# numeric cols
		feature_columns = []
		feature_numeric=[
				'round', 
				'HTGS', 'HTGC', 'HTP', "HTGD", 
				'ATGS', 'ATGC', 'ATP', "ATGD",
				
				'homeWin_halftime','awayWin_halftime',
				
				'pred_match_winner_1','pred_match_winner_2','pred_match_winner_N','pred_match_winner_1N','pred_match_winner_N2',
				
				'HTF', 'ATF', 
				'pred_under_over', 'pred_goals_home', 'pred_goals_away',
				
				#'HTPlayerDScore','HTPlayerMScore','HTPlayerFScore','ATPlayerDScore','ATPlayerMScore','ATPlayerFScore',
				
				'BW1', 'BWN', 'BW2',
				'BOver1_5goals', 'BUnder1_5goals','BOver2_5goals', 'BUnder2_5goals','BOver3_5goals', 'BUnder3_5goals',
				'BLDEM_YES', 'BLDEM_NO',
				
				'W1', 'WN', 'W2', "DiffWin12", "DiffWin1N", "DiffWin2N",
				'Over0_5goals', 'Over1_5goals', 'Over2_5goals', 'Over3_5goals', 'Over4_5goals', 
				'LDEM_YES'
				
		]
		feature_numeric = feature_numeric + feature_numeric_player_stats
		for keys in feature_numeric:
				feature_columns.append(feature_column.numeric_column(keys))

		# categorical indicator cols
		features_categorical=[
				'country_name',  'league_name',  'homeTeam',  'awayTeam'
		]
		#features_categorical=[]

		for keys in features_categorical:
				list=dataframe[keys].unique().tolist()
				tmp = feature_column.categorical_column_with_vocabulary_list(
						key=keys,
						vocabulary_list=list)
				thal_one_hot = feature_column.indicator_column(tmp)
				feature_columns.append(thal_one_hot)

		return feature_columns

def stackSplitData(dataframe, label_name):
		###########################################################
		#                 SPLIT THE DATA  -> Level0_df, Level1_df
		###########################################################
		print("******** Stacking splits : L0 & L1:leave One out   *******")
		#shuffle=true = <leave one out for stacking level1 >
		X_level0, X_level1, y_level0, y_level1 = train_test_split(
																									dataframe.drop([label_name],axis=1)
																								, dataframe[label_name]
																								, train_size = 0.33
																								, random_state = 2
																								, stratify = dataframe[label_name])

		#--Level1 Data
		print("Level1 Data shape...",X_level1.shape)
		level1_df=X_level1
		level1_df[label_name] = y_level1
		valid_df = level1_df
		valid_df = valid_df.reset_index(drop=True) 

		#--Level0 Data
		print("Level0 Data shape...",X_level0.shape)
		level0_df = X_level0
		level0_df[label_name] = y_level0

		#-- 
		print("Level0 Data Sub Splits:")
		level0_train_df, level0_test_df = StarifySplitData(level0_df,label_name, 0.15)

		return level0_train_df, level0_test_df, valid_df

#####################################################
def GetCSVtoDF(path):
		req = Request(path)
		req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
		content = urlopen(req)
		return pd.read_csv(content,sep=",", encoding = "ISO-8859-1",index_col=False,low_memory=False)

def getDataToTrain(version, download=False,
		feature_numeric_player_stats = [
				'HTPlayerGScore',
				'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
				'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
				'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
				'ATPlayerGScore',
				'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
				'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
				'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score']):
	
		inputdataframe =[]    
		current_time, current_display_time, current_timestamp = nowtime()
		#Training Dataset
		if download:
				path="https://web-concepts.fr/soccer/win-pronos/pronos-R-data-dev.php?action=training_dataset"
				inputdataframe = GetCSVtoDF(path)

		else:
				inputdataframe = pd.read_csv("training_dataset_fixtures-" + version + ".csv", sep=",", encoding = "ISO-8859-1", index_col=False, low_memory=False)
		
		print("******************************************")
		print("Imported data shape: ",inputdataframe.shape)
		# Filter coutries
		inputdataframe = inputdataframe[inputdataframe["country_code"] !='AR']
		inputdataframe = inputdataframe[inputdataframe["country_code"] !='GR']
		inputdataframe = inputdataframe[inputdataframe["country_code"] !='BE']

		print("******************************************")
		print("Imported data shape: ",inputdataframe.shape)

		dataframeInput=inputdataframe.copy()

		if(feature_numeric_player_stats!=[]):
			print("Scal Players Rating to RatingRange")
			for col in feature_numeric_player_stats:
					dataframeInput[col] = dataframeInput.apply(ratingRanges_v3, args=[col], axis=1); print(col," ok") ;
		#print("Players Rating scaled to RatingRange")

		print("  -> import dataset : ok", len(dataframeInput), " matches at ",current_time )
		return dataframeInput

def getDataToValidate(feature_numeric_player_stats = [
				'HTPlayerGScore',
				'HTPlayerD1Score','HTPlayerD2Score','HTPlayerD3Score','HTPlayerD4Score','HTPlayerD5Score','HTPlayerD6Score',
				'HTPlayerM1Score','HTPlayerM2Score','HTPlayerM3Score','HTPlayerM4Score','HTPlayerM5Score','HTPlayerM6Score','HTPlayerM7Score',
				'HTPlayerF1Score','HTPlayerF2Score','HTPlayerF3Score','HTPlayerF4Score',
				'ATPlayerGScore',
				'ATPlayerD1Score','ATPlayerD2Score','ATPlayerD3Score','ATPlayerD4Score','ATPlayerD5Score','ATPlayerD6Score',
				'ATPlayerM1Score','ATPlayerM2Score','ATPlayerM3Score','ATPlayerM4Score','ATPlayerM5Score','ATPlayerM6Score','ATPlayerM7Score',
				'ATPlayerF1Score','ATPlayerF2Score','ATPlayerF3Score','ATPlayerF4Score']):

		inputdataframe =[]  
		current_time, current_display_time, current_timestamp = nowtime()

		path="https://web-concepts.fr/soccer/win-pronos/pronos-R-data-dev.php?action=validation_dataset"
		inputdataframe = GetCSVtoDF(path)
		print("******************************************")
		print("Imported data shape: ",inputdataframe.shape)

		dataframeInput=inputdataframe.copy()

		for col in feature_numeric_player_stats:
				dataframeInput[col] = dataframeInput.apply(ratingRanges_v3, args=[col], axis=1); #print(col," ok") ;
		#print("Players Rating scaled to RatingRange")

		current_time, current_display_time, current_timestamp = nowtime()
		print("  -> import dataset : ok", len(dataframeInput), " matches at ",current_time )
		return dataframeInput

def getDataToPredict(offset, date):
		
		current_time, current_display_time, current_timestamp = nowtime()

		path = "https://web-concepts.fr/soccer/win-pronos/pronos-R-data-dev.php?action=predict_dataset&offset="+str(offset)+"&startDate="+str(date)
		inputdataframe = GetCSVtoDF(path)

		print("******************************************")
		print("Imported data shape: ",inputdataframe.shape)

		try:    
				inputdataframe = inputdataframe.sort_values(by=['event_timestamp']).reset_index(drop=True)
		except:
				r=requests.get(path, headers=headers)
				print(path)
				display(HTML(r.text))
				
		
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
		df=inputdataframe.copy()

		for col in PlayersRatingcols:
				df[col] = df.apply(ratingRanges_v3, args=[col], axis=1); #print(col," ok") ;
		#print("Players Rating scaled to RatingRange")

		print("  -> import dataset : ok", len(df), " matches at ",current_time )
		#print("  -> import dataset : ok", len(df), " matches at ",current_time , " with shape: ",df.shape)


		return df

def getPlayersRatings(importplayers=False, date="", offset=0):
		
		if(importplayers) :
				print("")
				print("#### Create the player dataset ...")
				url0="https://web-concepts.fr/soccer/update_fixtures_live.php"
				r0=requests.get(url0, headers=headers) 
				if(r0.status_code==200): print("  -> Update live fixtures : ok")

				url="https://web-concepts.fr/soccer/updateDatasetPlayers.php"
				if(date!="") :
						url="https://web-concepts.fr/soccer/updateDatasetPlayers.php?offset="+str(offset)+"&startDate="+str(date)
				r=requests.get(url, headers=headers) 
				if(r.status_code==200): print("  -> player dataset created: ok")
						
		# Make Player prediction for Dataset
		ppr.predictPlayerRating(startDate=date)

#####################################################
 
def resample(types,df, label_name):
		if types=="under":
				if(label_name=="FTR"):
						# Class count
						count_class_0, count_class_1, count_class_2 = df["FTR"].value_counts()

						# Divide by class
						df_class_0 = df[df["FTR"] == 0]
						df_class_1 = df[df["FTR"] == 1]
						df_class_2 = df[df["FTR"] == 2]
						
						df_class_1_under = df_class_1.sample(count_class_2)
						dfout = pd.concat([df_class_0, df_class_1_under, df_class_2], axis=0)
				else:
						# Class count
						count_class_0, count_class_1 = df[label_name].value_counts()

						# Divide by class
						df_class_0 = df[df[label_name] == 0]
						df_class_1 = df[df[label_name] == 1]
						
						df_class_0_under = df_class_0.sample(count_class_1)
						dfout = pd.concat([df_class_0, df_class_1_under], axis=0)
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
				
		dfout[label_name].value_counts().plot(kind='bar', title='Count (target)');
		
		return dfout

def col_ins(ds, var):
	# column names to indices
	return [ds.columns.get_loc(col) for col in var]

def smotenc(X, y, cat_var_ins):
	sm = SMOTENC(random_state=42, categorical_features=cat_var_ins)
	return sm.fit_sample(X, y)

def df_smotenc(df, dep_var, cat_var):
	y = df[dep_var]
	X = df.drop(dep_var, axis=1)
	cat_var_ins = col_ins(X, cat_var)
	
	# smotenc
	X_res, y_res = smotenc(X, y, cat_var_ins)
	print("x_train len: ",len(X))
	print("X_oversample len: ",len(X_res))
	
	# back to DataFrame (SMOTENC uses numpy)
	X_res = pd.DataFrame(X_res, columns=X.columns)
	y_res = pd.DataFrame(y_res, columns=[dep_var])
	df_res = y_res.merge(X_res, left_index=True, right_index=True)

	# set dtypes (which are lost when SMOTENC uses numpy)
	df_res = df_res.astype((df.dtypes))

	return df_res

def smotenc_upsample(train_df, label_name):
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
		print("SMOTENC over sampling")
		return df_smotenc(train_df, label_name, features_categorical)
		


		

