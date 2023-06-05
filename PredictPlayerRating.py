
import importlib

import libraries 
importlib.reload(libraries)
from libraries import *

#--
defaultModelVersion="v5"
#####################################################
###########       DATA CLEANING
#####################################################

def splitData(dataframe,label_name,splitratio,prints=True,encode=True):
		
		lab_enc = preprocessing.LabelEncoder()
		
		### Set the label : NextTrend
		TrendX_all = dataframe.drop([label_name],1)
		TrendY_all = dataframe[label_name]

		if(encode==True):
				TrendY_all = lab_enc.fit_transform(TrendY_all)

		X_train=[];  X_test=[];  y_train=[];  y_test=[]; 
		
		#### Split data to Training/Test
		testSize=int(len(dataframe)*splitratio)
		try:
				X_train, X_test, y_train, y_test = train_test_split(TrendX_all, TrendY_all,  test_size = testSize,  random_state = 2, stratify = TrendY_all)
		except:
				print("Warning: ",label_name," Not Stratified")
				X_train, X_test, y_train, y_test = train_test_split(TrendX_all, TrendY_all,  test_size = testSize, random_state = 2)
				
		#----
		if prints==True:
				print(len(X_train), 'train examples')
				print(len(X_test) , 'validation examples')

		return TrendX_all,TrendY_all,X_train, X_test, y_train, y_test,lab_enc

def stackSplitData(dataframe, label_name,features=["NextRating","NextRatingDiffAvgOv15","NextRatingDiffOv15",
						"NextRatingDiffAvgOv10","NextRatingDiffOv10",
						"NextRatingDiffOv07","NextRatingDiffOv02",
						"NextRatingDiffAvgOv07",
						"NextTrend",
						"NextTrendRng",
						"NextRatingDiffAvgOv02",
						"NextRatingDiffOv03",
						"NextRatingDiffAvgOv05","NextRatingDiffOv05",
						"NextRatingDiffAvgOv03",
						"NextRngOvAvg","NextOvAvgRng",
						'NextOvAvg','NextRngOvAvgRng'
						,'NextRatingRange']):

	dataframe=dataframe.drop(["countplayes","sameplayer"],1, errors='ignore')

	tmp=dataframe[label_name]        
	dataframe = dataframe.drop(features,axis=1)
	dataframe[label_name]=tmp;display(dataframe.head(1));


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

def StarifySplitData(dataframe, label_name, splitratio):

	TrendY_all = dataframe[label_name];
	
					
	#Split the dataframe into train, validation
	#----
	TrendX_all = dataframe.drop([label_name],axis=1);

	#----
	X_train, X_test, y_train, y_test = train_test_split(TrendX_all, TrendY_all,  test_size = int(len(dataframe)*splitratio),  random_state = 2, stratify = TrendY_all)

	train_df = X_train
	train_df[label_name] = y_train

	test_df = X_test
	test_df[label_name] = y_test

	print(len(train_df), 'train examples')
	print(len(test_df) , 'validation examples')

	return train_df, test_df

def createNewTargets(ds):    
		#Create New Targets
		print("")
		display(len(ds))
		print("Classify by ranges");
		ds['RatingRange'] = ds.apply(ratingRanges, args=['rating'], axis=1);print("feature 'RatingRange' added");
		ds['PrevRatingRange'] = ds.apply(ratingRanges, args=['PrevRating'], axis=1);print("feature 'PrevRatingRange' added");
		ds['PrevRating2Range'] = ds.apply(ratingRanges, args=['PrevRating2'], axis=1);print("feature 'PrevRating2Range' added");
		ds['avgRange'] = ds.apply(ratingRanges, args=['avg'], axis=1);print("feature 'avgRange' added");
		ds['avg2Range'] = ds.apply(ratingRanges, args=['avg2'], axis=1);print("feature 'avg2Range' added");
		ds['NextRatingRange'] = ds.apply(ratingRanges_v3, args=['NextRating'], axis=1);print("feature 'NextRatingRange' added");
		
		print("")
		print("Classify Trends");
		ds['NextTrend'] = ds.apply(trnd, axis=1).astype(int);print("feature 'NextTrend' added");
		ds['NextTrendRng'] = ds.apply(TrendRng, axis=1).astype(int);print("feature 'NextTrendRng' added");

		print("")
		print("Classify by positionning vs Avg");
		ds['NextOvAvg'] = ds.apply(OvAvg, axis=1).astype(int);print("feature 'NextOvAvg' added");
		ds['NextRngOvAvg'] = ds.apply(RngOvAvg, axis=1).astype(int);print("feature 'RngOvAvg' added");
		ds['NextRngOvAvgRng'] = ds.apply(RngOvAvgRng, axis=1).astype(int);print("feature 'RngOvAvgRng' added");
		ds['NextOvAvgRng'] = ds.apply(OvAvgRng, axis=1).astype(int);print("feature 'OvAvgRng' added");

		print("")
		print("Classify by NextRating distances to rating");
		ds['NextRatingDiffOv15'] = ds.apply(RatingDiffOv15, axis=1).astype(int); print("feature 'NextRatingDiffOv15' added");
		ds['NextRatingDiffOv10'] = ds.apply(RatingDiffOv10, axis=1).astype(int); print("feature 'NextRatingDiffOv10' added");
		ds['NextRatingDiffOv07'] = ds.apply(RatingDiffOv02, axis=1).astype(int); print("feature 'NextRatingDiffOv07' added");
		ds['NextRatingDiffOv05'] = ds.apply(RatingDiffOv05, axis=1).astype(int); print("feature 'NextRatingDiffOv05' added");
		ds['NextRatingDiffOv03'] = ds.apply(RatingDiffOv03, axis=1).astype(int); print("feature 'NextRatingDiffOv03' added");
		ds['NextRatingDiffOv02'] = ds.apply(RatingDiffOv02, axis=1).astype(int); print("feature 'NextRatingDiffOv02' added");

		print("")
		print("Classify by NextRating distances to Avg");
		ds['NextRatingDiffAvgOv15'] = ds.apply(RatingDiffAvgOv15, axis=1).astype(int); print("feature 'NextRatingDiffAvgOv15' added");
		ds['NextRatingDiffAvgOv10'] = ds.apply(RatingDiffAvgOv10, axis=1).astype(int); print("feature 'NextRatingDiffAvgOv10' added");
		ds['NextRatingDiffAvgOv07'] = ds.apply(RatingDiffAvgOv07, axis=1).astype(int); print("feature 'NextRatingDiffAvgOv07' added");
		ds['NextRatingDiffAvgOv05'] = ds.apply(RatingDiffAvgOv05, axis=1).astype(int); print("feature 'NextRatingDiffAvgOv05' added");
		ds['NextRatingDiffAvgOv03'] = ds.apply(RatingDiffAvgOv03, axis=1).astype(int); print("feature 'NextRatingDiffAvgOv03' added");
		ds['NextRatingDiffAvgOv02'] = ds.apply(RatingDiffAvgOv02, axis=1).astype(int); print("feature 'NextRatingDiffAvgOv02' added");
		
		return ds

def prepareData(ds, Predict=False):
		dataframe=ds.copy()

		#Convert to float
		dataframe["LastTrend"]    = pd.to_numeric(dataframe["LastTrend"], errors='coerce')
		dataframe["CumulTrend"]   = pd.to_numeric(dataframe["CumulTrend"], errors='coerce')
		dataframe["NextRating"]   = pd.to_numeric(dataframe["NextRating"], errors='coerce')
		dataframe["age"]          = pd.to_numeric(dataframe["age"], errors='coerce')
		dataframe["height"]       = pd.to_numeric(dataframe["height"], errors='coerce')
		dataframe["weight"]       = pd.to_numeric(dataframe["weight"], errors='coerce')

		#Filter the data
		if(Predict==False):dataframe=dataframe[dataframe["event_date"]<="2020-12-01"]
		if(Predict==False):dataframe=dataframe[dataframe["NextRating"]>0]
		
		dataframe = dataframe[dataframe["avg"]>0]
		dataframe = dataframe[dataframe["round"] >2]
		dataframe = dataframe[dataframe["season"] >=2018]
		dataframe = dataframe[dataframe["NextRating"] !=3.5]
		dataframe = dataframe[dataframe["NextRating"] !=3.7]
		

		#Drop unused data
		dataframe=dataframe.drop(["event_date"
															,"H2Hid","fixture_id","country_code","team2_id","Nextteam2Id"], axis=1)
		
		#Clean Nan
		dataframe[["position","birth_country"]] = dataframe[["position","birth_country"]].fillna('Unknown')
		
		#Label encode string features
		dataframe["team1"] = preprocessing.LabelEncoder().fit_transform(dataframe["team1"])
		dataframe["team2"] = preprocessing.LabelEncoder().fit_transform(dataframe["team2"])
		dataframe["Nextteam2"] = preprocessing.LabelEncoder().fit_transform(dataframe["Nextteam2"])
		dataframe["position"] = preprocessing.LabelEncoder().fit_transform(dataframe["position"])
		dataframe["birth_country"] = preprocessing.LabelEncoder().fit_transform(dataframe["birth_country"])
		
		#display(dataframe.describe())
		
		#Clean Nan
		teamStatCol=["avg","avg2","std2","varSTD","varSTDdiff", "varSTDratio",
							 'Nextfixture_id',
							 'rating',"PrevRating2",
							 'teamPointsTo','teamPointsFrom','teamPoints',
							 'age','height','weight']
		
		dataframe[teamStatCol] = dataframe[teamStatCol].fillna(0)
		
		teamAvgCol=["team1Gavg","team1Mavg","team1Favg","team1Davg",
							 "team2Gavg","team2Mavg","team2Favg","team2Davg",
							 "Nextteam2Gavg","Nextteam2Mavg","Nextteam2Favg","Nextteam2Davg"]
		
		dataframe[teamAvgCol]=dataframe[teamAvgCol].fillna(0)
		
		
		H2HCol = ["LH2HFTR","LH2Hteam1Mavg","LH2Hteam1Gavg","LH2Hteam1Favg","LH2Hteam1Davg",
								"LH2Hteam2Mavg","LH2Hteam2Gavg","LH2Hteam2Favg","LH2Hteam2Davg","LH2Hrating",
								"AvgH2Hteam1Mavg","AvgH2Hteam1Gavg","AvgH2Hteam1Favg","AvgH2Hteam1Davg",
								"AvgH2Hteam2Mavg","AvgH2Hteam2Gavg","AvgH2Hteam2Favg","AvgH2Hteam2Davg","AvgH2Hrating"]
		
		dataframe[H2HCol]=dataframe[H2HCol].fillna(0)
		
		specialCol=["OverAvg","OverAvg2","LastTrend","varSTDSign"]
		dataframe[specialCol]= dataframe[specialCol].fillna(-9999)

		dataframe["PrevRating"]  = np.where( pd.isnull(dataframe["PrevRating"]), dataframe["rating"],  dataframe["PrevRating"])
		dataframe["PrevRating2"] = np.where( pd.isnull(dataframe["PrevRating2"]), dataframe["PrevRating"], dataframe["PrevRating2"])
		dataframe["LH2Hrating"]  = np.where( pd.isnull(dataframe["LH2Hrating"]), dataframe["PrevRating"], dataframe["LH2Hrating"])
		dataframe["CumulTrend"]  = np.where( pd.isnull(dataframe["CumulTrend"]), dataframe["LastTrend"], dataframe["CumulTrend"])
		

		print("drop rows:",len(dataframe[dataframe.isnull().any(axis=1)] ))
		#display()
		#dataframe = dataframe.dropna()
		#display(dataframe.describe())


		#Reformat data types
		dataframe[['varSTDSign',
							 'OverAvg','OverAvg2',
							 'LastTrend','CumulTrend']]=dataframe[['varSTDSign',
																										 'OverAvg','OverAvg2',
																										 'LastTrend','CumulTrend']].astype(int)
		
		dataframe[['Nextfixture_id',
							 'teamPointsTo','teamPointsFrom',
							 'teamPoints']]=dataframe[['Nextfixture_id',
																				 'teamPointsTo','teamPointsFrom',
																				 'teamPoints']].astype(int)

		dataframe[['age','height','weight']]=dataframe[['age','height','weight']].astype(int)
		
		#Filter noisy data
		dataframe = dataframe[dataframe["PrevRating"] !=0]
		dataframe = dataframe[dataframe["rating"] !=0]
		
		
		######  Normalize data #####
		scl = MinMaxScaler() ; # MinMaxScaler() ; StandardScaler() ; RobustScaler()

		ScalCols = ["teamPointsTo","teamPointsFrom","teamPoints",
								"minutes_played","cards_yellow","cards_red",
								"round","maxRound","roundToPlay",
								"season",
								"age","height","weight",
								"avg", "avg2", "PrevRating2", "PrevRating", "rating",
								"std2","varSTD","varSTDdiff","varSTDratio",
								"team1Mavg", "team1Gavg", "team1Favg", "team1Davg",
								"team2Mavg", "team2Gavg", "team2Favg", "team2Davg",
								"Nextteam2Mavg", "Nextteam2Gavg", "Nextteam2Favg", "Nextteam2Davg",
								"RatingRange", "PrevRatingRange", "PrevRating2Range", "avgRange", "avg2Range",
								"LH2HFTR","LH2Hteam1Mavg","LH2Hteam1Gavg","LH2Hteam1Favg","LH2Hteam1Davg",
								"LH2Hteam2Mavg","LH2Hteam2Gavg","LH2Hteam2Favg","LH2Hteam2Davg","LH2Hrating",
								"AvgH2Hteam1Mavg","AvgH2Hteam1Gavg","AvgH2Hteam1Favg","AvgH2Hteam1Davg",
								"AvgH2Hteam2Mavg","AvgH2Hteam2Gavg","AvgH2Hteam2Favg","AvgH2Hteam2Davg","AvgH2Hrating"
								,"OverAvg","OverAvg2","LastTrend","CumulTrend"
								,"birth_country"
								,"position"
							 ]
		 
		ScalCols = ["teamPointsTo","teamPointsFrom","teamPoints"
								,"minutes_played","cards_yellow","cards_red"
								,"round","maxRound","roundToPlay"
								,"season"
								,"age","height","weight"
								,"std2","varSTD","varSTDdiff","varSTDratio"
								,"birth_country"
								,"position"
							 ]

		#dataframe[ScalCols] = scl.fit_transform(dataframe[ScalCols])
		
		print("dataframe shape:",dataframe.shape)
		return dataframe

def exportPredictionsToDb(dfTemp):
		#display(dfTemp.shape)
		uniqueIds=dfTemp["Nextfixture_id"].unique()

		for index, row in dfTemp.iterrows():
				#print("Update dataset predictions",int(row["Nextfixture_id"])," with ",str(row["NextRatingRange"]) )
				url="https://web-concepts.fr/soccer/win-pronos/pronos-R-data-dev.php"
				param={
						"action":"savePlayerPrediction"
						,"Nextfixture_id" :(int(row["Nextfixture_id"]))
						,"player_id"      :(int(row["player_id"]))
						,"NextRating"     :str(row["NextRatingRange"])
				}
				#print(param)
				r=requests.get(url,params=param, headers=headers) 
				
				#if(r.status_code==200): print("    -> ",len(dfTemp)," player predictions save: ok")
				
				#display(HTML(r.text))
				#display("->dataset player rating updated")
		#print("->dataset with player rating updated : ")

#####################################################
##############  IMPORT DATA 
#####################################################

def importdatas(version=defaultModelVersion):
		
		df0A=pd.read_csv("training_dataset_players-" + version + ".csv", sep=",", encoding = "ISO-8859-1",index_col=False,low_memory=False)

		df0A = df0A[df0A["country_code"] !='AR']
		df0A = df0A[df0A["country_code"] !='GR']
		df0A = df0A[df0A["country_code"] !='BE']

		#########################################
		#########################################

		ds=df0A.copy()
		print(df0A.shape)

		display(ds.head())

		###################################
		######  Add New features      #####
		ds = createNewTargets(ds)


		###################################
		######  Filter and clean data #####
		df = prepareData(ds)
		display(df.head())

		###################################
		######  Save To CSV clean data #####

		df.to_csv("training_full_dataset_players-" + version + ".csv", index=False, sep=",", encoding = "ISO-8859-1")
		return df

#####################################################
###########      DATAFRAME RULES           
#####################################################

def ratingRanges(x,col):
				if (x[col] >= 3 and x[col] < 3.25)  :
						return 3
				elif (x[col] > 3.25 and x[col] < 3.75)  :
						return 3.5
				elif (x[col] > 3.75 and x[col] < 4.25)  :
						return 4
				elif (x[col] > 4.25 and x[col] < 4.75)  :
						return 4.5
				elif (x[col] > 4.75 and x[col] < 5.25 ) :
						return 5
				elif (x[col] > 5.25 and x[col] < 5.75)  :
						return 5.5
				elif (x[col] > 5.75 and x[col] < 6.25 ) :
						return 6
				elif (x[col] > 6.25 and x[col] < 6.75)  :
						return 6.5
				elif (x[col] > 6.75 and x[col] < 7.25)  :
						return 7
				elif (x[col] > 7.25 and x[col] < 7.75)  :
						return 7.5
				elif (x[col] > 7.75 and x[col] < 8.25)  :
						return 8
				elif (x[col] > 8.25 and x[col] < 8.75)  :
						return 8.5
				elif (x[col] > 8.75 and x[col] < 9.25)  :
						return 9
				elif (x[col] > 9.25 and x[col] < 9.75)  :
						return 9.5
				elif (x[col] > 9.75) :
						return 10
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

def trnd(x):
		if (x['NextRating'] - x['rating'] > 0) :
				return 1
		elif(x['NextRating'] - x['rating'] < 0) :
				return -1
		else: return 0

def TrendRng(x):
		if (x['NextRatingRange'] - x['RatingRange'] > 0) :
				return 1
		elif(x['NextRatingRange'] - x['RatingRange'] < 0) :
				return -1
		else: return 0

def OvAvg(x):
		if (x['NextRating'] - x['avg'] >= 0) :  return 1 
		else: return 0
def RngOvAvg(x):
		if (x['NextRatingRange'] - x['avg'] >= 0) :  return 1
		else: return 0
def RngOvAvgRng(x):
		if (x['NextRatingRange'] - x['avgRange'] >= 0) : return 1
		else: return 0
def OvAvgRng(x):
		if (x['NextRating'] - x['avgRange'] >= 0) : return 1
		else: return 0

def RatingDiffOv10(x):
		if  (abs(x['NextRating'] - x['rating']) >= 1) : return 1
		else: return 0
def RatingDiffOv05(x):
		if (abs(x['NextRating'] - x['rating']) >= 0.5) : return 1
		else: return 0
def RatingDiffOv15(x):
		if (abs(x['NextRating'] - x['rating']) >= 1.5) : return 1
		else: return 0
def RatingDiffOv03(x):
		if (abs(x['NextRating'] - x['rating']) >= 0.3) : return 1
		else: return 0
def RatingDiffOv02(x):
		if (abs(x['NextRating'] - x['rating']) >= 0.2) : return 1
		else: return 0
def RatingDiffOv07(x):
		if (abs(x['NextRating'] - x['rating']) >= 0.7) : return 1
		else: return 0
		
def RatingDiffAvgOv02(x):
		if (abs(x['NextRating'] - x['avg']) >= 0.2) : return 1
		else: return 0
def RatingDiffAvgOv03(x):
		if (abs(x['NextRating'] - x['avg']) >= 0.3) : return 1
		else: return 0
def RatingDiffAvgOv05(x):
		if (abs(x['NextRating'] - x['avg']) >= 0.5) : return 1
		else: return 0
def RatingDiffAvgOv07(x):
		if (abs(x['NextRating'] - x['avg']) >= 0.7) : return 1
		else: return 0
def RatingDiffAvgOv10(x):
		if (abs(x['NextRating'] - x['avg']) >=1) : return 1
		else: return 0
def RatingDiffAvgOv15(x):
		if (abs(x['NextRating'] - x['avg']) >=1.5) : return 1
		else: return 0

#####################################################
###########      MACHINE LEARNING TRAINING        
#####################################################
def predictPlayerRating(startDate=""):
		
		
		print("#### Make Player prediction for Dataset") 
		#Import Predict data - fixture Ids
		#--
		path="https://web-concepts.fr/soccer/win-pronos/pronos-R-data-dev.php?action=predict_player_dataset&startDate="+str(startDate)
		predict_dataframeInput=pd.read_csv(path, sep=",", encoding = "ISO-8859-1")
		
		print(" -> Import the player dataset for prediction ", len(predict_dataframeInput), " players")
		#display(len(predict_dataframeInput))
		if(len(predict_dataframeInput)>0):
		
				ds = predict_dataframeInput.copy()
				###################################
				######  Filter and clean data #####
				###################################
				print(" -> Process the player dataset for prediction")
				ds = createNewTargets(ds)
				display(ds.shape)
				df = prepareData(ds,Predict=True)
				display(df.shape)

				#display(df.tail())

				
				print(" -> Make the player predictions for ",len(df), " players")
				if(len(df)>0):
					features=["NextRatingDiffAvgOv15","NextRatingDiffOv15",
										"NextRatingDiffAvgOv10","NextRatingDiffOv10",
										"NextRatingDiffOv07","NextRatingDiffOv02",
										"NextRatingDiffAvgOv07",
										"NextTrend",
										"NextTrendRng",
										"NextRatingDiffAvgOv02",
										"NextRatingDiffOv03",
										"NextRatingDiffAvgOv05","NextRatingDiffOv05",
										"NextRatingDiffAvgOv03",
										"NextRngOvAvg","NextOvAvgRng",
										'NextOvAvg','NextRngOvAvgRng'
									 ,'NextRatingRange'
									 ,'NextRating']


					directory='trainingModels/model-pr-'
					dfTempo=df.copy()

					#print("")
					#print("Build dataset : features to keep/add ")
					for f2 in features:
							dfTempo=dfTempo.drop([f2],1)
					#print("dataset init shape:",dfTempo.shape)

					reset_random_seeds()
					for f2 in features:
							filename       = directory+f2+'.sav'
							encoderfilename= directory+'encoder-'+f2+'.npy'
							#load model
							load_model = pickle.load(open(filename, 'rb'))

							#make prediction

							predic = load_model.predict(dfTempo)

							#decode prediction
							if (f2=="NextRatingRange" or f2=="NextRating" ):
									labl_enc = LabelEncoder()
									labl_enc.classes_ = np.load(encoderfilename)
									predic = labl_enc.inverse_transform(predic)

							#append prediction to dataset
							dfTempo[f2] = predic
							#print("add ",f2, " ",dfTempo.shape)


					#save data in dB
					print(" -> Update dataset with player predictions")
					exportPredictionsToDb(dfTempo)

					return dfTempo

		

		else:
				print("Player dataset empty")
				return False

def FileNames(label_name, version=defaultModelVersion):
		
		directory       = 'trainingModels/model-pr-'
		version ="-" + version
				
		filename        = directory + label_name + version +'.sav'
		encoderfilename = directory +'encoder-'  + label_name + version +'.npy'
		return filename,encoderfilename

def evaluateMLModel(
		label_name, 
		train_df, 
		test_df, 
		valid_df, 
		version,
		model_type="RFC"
		):
	level1_MLpred_df, RF_clf = trainMLModel( label_name, train_df, test_df, valid_df, version, "evaluate", model_type=model_type,
		predictProba=True )
	return level1_MLpred_df, RF_clf

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


	filename,encoderfilename = FileNames(label_name,version)

	#--
	x_valid = valid_df.drop([label_name],axis=1)
	y_valid = valid_df[label_name]

		#--
	if(train=="train" or train=="evaluate"):
		print("")
		print("==> Start Training - ", label_name) 
		reset_random_seeds()
		#Unbalanced - resampling
		#if(label_name=="Team2Win"):
		#    train_df = resample("over", train_df, label_name) #under,over
		#display(train_df.head())
		
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

			#-- model - Classifier & param_grid 
			if (model_type=="RFC"):

					level0_clf = RandomForestClassifier(random_state = 2 , n_estimators = 100, class_weight='balanced')
					param_grid = {
							#'max_features': range(1,Nbfeatures,1) ,
							#'max_features': [1,2,3] ,
							'max_depth' : range(8, 24, 1),
							'min_samples_leaf' : range(1, 5, 1) #range(1, 4, 1) [3,4,5]
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
			#if(label_name == "NextRatingDiffAvgOv15"):
			#    NbFolds = 10

			kfolds = StratifiedKFold(NbFolds)
			refit = 'recall_macro'

			print("GridSearchCV refit = ", refit)
			CV_rfc = GridSearchCV( estimator   = level0_clf
														, n_jobs     = -1
														, param_grid = param_grid
														, cv         = kfolds.split(x_train, y_train)
														, scoring    = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
														, verbose    = 10
														, refit      = refit )
			
			Nbfeatures=len(x_train. columns)
			print("Start CV Fit Nbfeatures: ",Nbfeatures)

			CV_rfc.fit( x_train, y_train )

			print ("best score: ", CV_rfc.best_score_ )
			print ("best params: ", CV_rfc.best_params_)

			#Create the model from best_estimators
			model = CV_rfc.best_estimator_

			Nbfeatures=len(x_train. columns)
			print("Start training Nbfeatures: ",Nbfeatures)

			model.fit(x_train, y_train)

			#Save the model
			print("")
			print("==> Save the model : ",filename)
			pickle.dump(model, open(filename, 'wb'))


		
				
		print(x_train.info());
		pred_train  = model.predict(x_train)
		pred_test   = model.predict(x_test)
		print(pred_train);

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
	if(label_name=="NextTrend" or label_name=="NextTrendRng"):
		predictions= pd.DataFrame( data = preds, columns=(['D_ML', 'O_ML', 'I_ML']) ) 
	else:
		predictions= pd.DataFrame( data = preds, columns=([label_name + '_ML','not_'+label_name + '_ML']) ).drop(['not_'+label_name + '_ML'],axis=1)

	predictions = predictions.reset_index(drop=True)

		#---
	level1_MLpred_df = pd.concat( [ valid_df['Nextfixture_id'],valid_df['player_id'],valid_df[label_name], predictions ] , axis=1)

	if(predictProba):
		return level1_MLpred_df , model
	else:
		#printt("test")
		tmpprediction = tmpprediction.reset_index(drop=True)
		return tmpprediction_class , model

def CVtrain(x_train, x_test, y_train, y_test,label_name,version, train=True):
		
	filename,encoderfilename = FileNames(label_name,version)

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
				#'n_estimators' : [100,150,200,250] ,
				'max_features': range(1,8,1) ,
				'max_depth' : range(15, 34, 1),
				'min_samples_leaf' : range(1, 4, 1)
		}


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

def trainModel(data, label_name, train="train", version=defaultModelVersion):
		
		filename,encoderfilename= FileNames(label_name, version)
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
						clf = CVtrain(x_train, x_test, y_train, y_test ,label_name,version, train=True)
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
	

def trainModelOld(df):
	## init parameters
	directory='trainingModels/lab/model-pr-'

	features=["NextRatingDiffAvgOv15","NextRatingDiffOv15",
						"NextRatingDiffAvgOv10","NextRatingDiffOv10",
						"NextRatingDiffOv07","NextRatingDiffOv02",
						"NextRatingDiffAvgOv07",
						"NextTrend",
						"NextTrendRng",
						"NextRatingDiffAvgOv02",
						"NextRatingDiffOv03",
						"NextRatingDiffAvgOv05","NextRatingDiffOv05",
						"NextRatingDiffAvgOv03",
						"NextRngOvAvg","NextOvAvgRng",
						'NextOvAvg','NextRngOvAvgRng'
					 ,'NextRatingRange']

	#features=["NextRatingDiffAvgOv15"]

	# Classification Training to get NextRatingRange
	for label_name in features:
		dfTempo=df.copy()
		#dfTempo=dfTempo.drop(["NextRating"], axis=1)

		print("")
		print("###### START ", label_name , "  #########")
		print(" dataframe shape ",dfTempo.shape)

		rem=0
		#replace in dataframe predictionsToInject exept for label to predict
		print("")
		print("Build dataset : features to keep/add ")
		# Drop all features
		for f2 in features:
			if (f2!=label_name):
				dfTempo=dfTempo.drop([f2],1)

		print("Nb of features init",dfTempo.shape)    
		#display(dfTempo.head())

		print("")
		#Load new features
		for f2 in features:

			version="-v4"
			Load_filename = directory+f2+version+'.sav'
			Load_encoderfilename= directory+'encoder-'+f2+version+'.npy'

			testAdd=0
			testRem=0

			if (f2!=label_name):
				testAdd=1
			if (f2==label_name):
				rem=1
				print(" = keep ",f2, " ", dfTempo.shape)
			if (rem==1 and f2!=label_name):
					testRem=1
			if(testAdd==1 and testRem==0):

				df2Predic=dfTempo.drop([label_name],1)
				
				#load model
				load_model = pickle.load(open(Load_filename, 'rb'))
				predic = load_model.predict(df2Predic)

				if (f2=="NextRatingRange" or f2=="NextRating" ):
					labl_enc = LabelEncoder()
					print('Load encoder ',f2)
					labl_enc.classes_ = np.load(Load_encoderfilename)
					predic = labl_enc.inverse_transform(predic)

				dfTempo[f2] = predic

				print(" + add ",f2, " ",dfTempo.shape," .... check prediction is equal than original:",df[f2].equals(dfTempo[f2]))

		print("")
		print("Nb of features for training",dfTempo.shape)      
		#display(dfTempo.head())

		#File name definition
		version="-v5"
		filename = directory+label_name+version+'.sav'
		encoderfilename= directory+'encoder-'+label_name+version+'.npy'

		#Split Data
		splitratio=0.05
		encode=False
		if (label_name=="NextRatingRange" or f2=="NextRating" ): encode=True
		print("")
		print("Split Data : training/validation")


		TrendX_all, TrendY_all, x_train, x_test, y_train, y_test, lab_enc = splitData(dfTempo
																																								,label_name
																																								,splitratio
																																								,encode=encode)

		#upsampling training dataset
		#smote = SMOTE(random_state=2)
		#X_oversample, y_oversample = smote.fit_sample(x_train, y_train)
		#print("x_train len: ",len(x_train))
		#print("X_oversample len: ",len(X_oversample))
		#x_train = X_oversample
		#y_train = y_oversample

		#display(TrendX_all)

		#save encoded
		if (label_name=="NextRatingRange" or label_name=="NextRating" ): 
				print("")
				print("==> Save encoders : ",encoderfilename)
				np.save(encoderfilename, lab_enc.classes_)


		#---
		##Train
		print("")
		print("==> Start Training - ", label_name) 

		reset_random_seeds()

		class_weights = dict(zip(np.unique(y_train), class_weight.compute_class_weight('balanced',
																						 np.unique(y_train),
																						 y_train))) 

		#display(y_train.value_counts())

		clf = RandomForestClassifier(
				n_jobs           = -1,
				random_state     = random.seed(1234),
				class_weight     = 'balanced')

		Nbfeatures=len(x_train. columns)
		param_grid = {  
				#'max_features': range(1,Nbfeatures,1) ,
				'max_depth' : range(14, 38, 1),
				'min_samples_leaf' : range(1, 3, 1)
		}
		param_grid = { 
				'max_depth' : range(14, 19, 1)
				, 'max_features': range(33, 39, 1)
		}

		cvTrain      = True
		cvFolds      = 10
		kfolds       = StratifiedKFold(5)

		#---
		if (label_name=="NextRatingDiffAvgOv15"): param_grid = {  'max_depth' : [14], 'max_features': [32] } ; cvTrain = True ;
		if (label_name=="NextRatingDiffOv15"):    param_grid = {  'max_depth' : [14], 'max_features': [32] } ; cvTrain = False ;
		if (label_name=="NextRatingDiffAvgOv10"): param_grid = {  'max_depth' : [14], 'max_features': [34] } ; cvTrain = False ;
		if (label_name=="NextRatingDiffOv10"):    param_grid = {  'max_depth' : [16], 'max_features': [34] } ; cvTrain = False ;
		if (label_name=="NextRatingDiffOv07"):    param_grid = {  'max_depth' : [17], 'max_features': [37] } ; cvTrain = False ;
		if (label_name=="NextRatingDiffOv02"):    param_grid = {  'max_depth' : [10], 'max_features': [30] } ; cvTrain = False ;
		if (label_name=="NextRatingDiffAvgOv07"): param_grid = {  'max_depth' : [15], 'max_features': [30] } ; cvTrain = False ;
		if (label_name=="NextTrend"):             param_grid = {  'max_depth' : [17], 'max_features': [33] } ; cvTrain = False ;
		if (label_name=="NextTrendRng"):          param_grid = {  'max_depth' : [17], 'max_features': [33] } ; cvTrain = False ;
		if (label_name=="NextRatingDiffAvgOv02"): param_grid = {  'max_depth' : [17], 'max_features': [37] } ; cvTrain = False ;
		if (label_name=="NextRatingDiffOv03"):    param_grid = {  'max_depth' : [17], 'max_features': [37] } ; cvTrain = False ;     
		if (label_name=="NextRatingDiffAvgOv05"): param_grid = {  'max_depth' : [17], 'max_features': [35] } ; cvTrain = False ;
		if (label_name=="NextRatingDiffOv05")   : param_grid = {  'max_depth' : [17], 'max_features': [38] } ; cvTrain = False ;
		if (label_name=="NextRatingDiffAvgOv03"): param_grid = {  'max_depth' : [17], 'max_features': [38] } ; cvTrain = False ;
		if (label_name=="NextRngOvAvg"):          param_grid = {  'max_depth' : [18], 'max_features': [34] } ; cvTrain = False ;

		if (label_name=="NextOvAvgRng"):          param_grid = {  'max_depth' : [17], 'max_features': [33] } ; cvTrain = False ;
		if (label_name=="NextOvAvg"):             param_grid = {  'max_depth' : [15], 'max_features': [38] } ; cvTrain = False ;
		if (label_name=="NextRngOvAvgRng"):       param_grid = {  'max_depth' : [18], 'max_features': [35] } ; cvTrain = False ;

		if (label_name=="NextRatingRange"):       param_grid = {  'max_depth' : [14], 'max_features': [35] } ; cvTrain = False ;


		if(cvTrain):
				
			CV_rfc = GridSearchCV(estimator = clf
														, n_jobs = 8
														, param_grid = param_grid
														, cv         = kfolds.split(x_train, y_train)
														, scoring    = ['accuracy','precision_macro','recall_macro', 'f1_macro']
														, verbose    = 10
														,refit       = 'f1_macro')

			CV_rfc.fit(x_train, y_train)

			print (CV_rfc.best_score_)
			print (CV_rfc.best_params_)

			clf=CV_rfc.best_estimator_
				
		else:
			print("loading saved model... ", filename) 
			clf = pickle.load(open(filename, 'rb'))

		print("==> Start Prediction - ", label_name) 

		pred_train  = clf.predict(x_train)
		accuracy_score_train = accuracy_score(y_train,pred_train)

		pred_test = clf.predict(x_test)
		accuracy_score_test = accuracy_score(y_test,pred_test)

		print("     ... Train - accuracy_score = %1.2f" % accuracy_score_train)
		print("     ... Test  - accuracy_score = %1.5f" % accuracy_score_test)
		print("     ... Test  - classification:\n",classification_report(y_test, pred_test))

				
		#Save the model
		print("")
		print("==> Save the model : ",filename, " ",dfTempo.shape )
		# save the model to disk
		pickle.dump(clf, open(filename, 'wb'))

		print("")
		print("Check Results:")

		##############    
		tmpload_model = pickle.load(open(filename, 'rb'))
		tmpprediction  = tmpload_model.predict(TrendX_all)


		if (label_name=="NextRatingRange" or label_name=="NextRating" ): 
			labl_enc = LabelEncoder()
			print('Load encoder ',label_name)
			labl_enc.classes_ = np.load(encoderfilename)
			TrendY_all    = labl_enc.inverse_transform(TrendY_all)
			tmpprediction = labl_enc.inverse_transform(tmpprediction)

		accuracy_score_full = accuracy_score(TrendY_all,tmpprediction)
		print("--------> Full - accuracy_score = %1.3f" % accuracy_score_full)

		display(pd.DataFrame({
				'Nextfixture_id':TrendX_all['Nextfixture_id']
				,'player_id':TrendX_all['player_id']
				,'Rating':TrendX_all["rating"]
				,'Avg':TrendX_all["avg"]
				, 'Actual': TrendY_all
				,'Predicted': tmpprediction.flatten()
		}).head())

		print("")
		print("###### END ",label_name, "  #########")
		
		

