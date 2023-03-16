import pandas as pd
from pathlib import Path
from datetime import datetime as dt, time

class DataHandler:
    """
    Datahandling class that gets the data and formats it
    """
    
    # gets the raw daily data and concatenates it into a single dataframe
    def read_raw(self, loc, start=None, end=None):
        
        # gets min and max timestamps if no data
        if start is None:
            start = pd.Timestamp.min
            
        if end is None:
            end = pd.Timestamp.max
            
        # getting intraday data
        intraday_data_list = []
        for file in sorted((loc / "intraday_data").resolve().glob("*.csv")):
            
            # checking if date of file is between start and end
            file_date = self._intraday_file2date(file)
            if file_date < start:
                continue
            if file_date > end:
                break
            
            intraday_data_list.append(pd.read_csv(file, parse_dates =["Date"]))
            
        intraday_data = pd.concat(intraday_data_list)
        
        # getting daily_data
        daily_data_list = []
        for file in sorted((loc / "daily_data").resolve().glob("*.csv")):
            
            # checking if date of file is between start and end
            file_date = self._daily_file2date(file)
            if file_date < start:
                continue
            if file_date > end:
                break
                
            daily_data_list.append(pd.read_csv(file, parse_dates =["Date"]))#, usecols=daily_cols))
            
        daily_data = pd.concat(daily_data_list)

        # merging intraday with daily data                              
        data = intraday_data.merge(daily_data, how="left", left_on=["Date", "Id"], right_on=["Date", "ID"])
        
        # setting index as datetime index
        data.index = data['Date'] + pd.to_timedelta(data['Time'])
        
        data.drop(["ID", "Date", "Time"], axis=1, inplace=True)
                                      
        return data.sort_index()
    
    # functions to translate file name to datetime
    def _intraday_file2date(self, file):
        return pd.to_datetime(file.stem)
    
    def _daily_file2date(self, file):
        return pd.to_datetime(file.stem[4:])
    
    # function to read processed data
    def read_processed(self, loc, start=None, end=None):

         # gets min and max timestamps if no data
        if start is None:
            start = pd.Timestamp.min
            
        if end is None:
            end = pd.Timestamp.max
        
        # getting daily_data
        daily_data_list = []
        for file in sorted(loc.glob("*.csv")):
            
            file_date = self._processed_file2date(file)
            if file_date < start:
                continue
            if file_date > end:
                break
            # getting Market Regime as category and renaming the multiindex and setting it again
            daily_data_list.append(pd.read_csv(file, parse_dates =[1], dtype={"Market Regime":"category"}).rename(columns={"Unnamed: 0": "Id", "Unnamed: 1": "Date"}).set_index(["Id", "Date"]))

        if len(daily_data_list) == 0:
            raise ValueError("features path has no csv files or no files in date interval")
        
        daily_data = pd.concat(daily_data_list).sort_index()

        # returning X, y, weights
        return daily_data.drop(["Sample Weights", "Target"], axis=1), daily_data["Target"], daily_data["Sample Weights"]

    # same as before
    def _processed_file2date(self, file):
        return pd.to_datetime(file.stem)
    
    # storing the processed data to daily files
    def store_dataset(self, out, dataset):
        for day, daily_data in dataset.groupby(level=1):
            path = (out / f"{dt.strftime(day, '%Y-%m-%d')}.csv").resolve()
            daily_data.to_csv(path)

    # storing the predictions
    def store_predictions(self, out, y_preds, y):
        predictions_df = pd.DataFrame(y_preds, index=y.index, columns=["Pred"]).reset_index(0)
        predictions_df["Time"] = time(15, 30)
        predictions_df[['Time', 'Id', 'Pred']].to_csv((out / "predictions.csv").resolve())


        
