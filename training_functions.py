
import general_functions

def prepareData(dataframeInput,label_name):
    
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
    
    
    
    # Add League avg success on first half
    gp=dataframe.groupby(['league_id'])
    dataframe["homeWin_halftime"] = gp['homeWin_halftime'].agg(np.mean)
    dataframe["awayWin_halftime"] = gp['awayWin_halftime'].agg(np.mean)
    
    
   
    
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
    dataframe=dataframe[dataframe["HTPlayerD1"]!="IDnull"]
    dataframe=dataframe[dataframe["HTPlayerM1"]!="IDnull"]
    dataframe=dataframe[dataframe["HTPlayerF1"]!="IDnull"]  
    
    dataframe=dataframe[dataframe["ATPlayerG"] !="IDnull"]
    dataframe=dataframe[dataframe["ATPlayerD1"]!="IDnull"]
    dataframe=dataframe[dataframe["ATPlayerM1"]!="IDnull"]
    dataframe=dataframe[dataframe["ATPlayerF1"]!="IDnull"]  
    
    
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
    
    
    if label_name !="FTR":
        dataframe = dataframe.drop(['FTR'] , axis=1)  
    
    ##----
    #Mélange les données
    dataframe = dataframe.reindex(np.random.permutation(dataframe.index))
    
    return dataframe


