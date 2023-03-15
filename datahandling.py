import pandas as pd
from pathlib import Path

class DataHandler:
    """
    Datahandling class that gets the data and formats it
    """
    def __init__(self, read_path=None):
        if read_path is None:
            self._read_path = (Path.cwd().parent / "data").resolve()
        else:
            self._read_path
        
    def read(self, start:str=None, end:str=None):
        
        if start is None:
            start = pd.Timestamp.min
        else:
            start = pd.to_datetime(start)
            
        if end is None:
            end = pd.Timestamp.max
        else:
            end = pd.to_datetime(end)
            
            
        #intraday_cols = ["Date", "Time", "Id", "CumReturnResid", "CumVolume"]
        
        intraday_data_list = []
        for file in sorted((self._read_path / "intraday_data").resolve().glob("*.csv")):
            
            file_date = self._intraday_file2date(file)
            if file_date < start:
                continue
            if file_date > end:
                break
                
            intraday_data_list.append(pd.read_csv(file, parse_dates =["Date"]))#, usecols=intraday_cols))
            
        intraday_data = pd.concat(intraday_data_list)
        
        #daily_cols = ["Date", "ID", "EST_VOL", "MDV_63", "Volume"]
        
        daily_data_list = []
        for file in sorted((self._read_path / "daily_data").resolve().glob("*.csv")):
            
            file_date = self._daily_file2date(file)
            if file_date < start:
                continue
            if file_date > end:
                break
                
            daily_data_list.append(pd.read_csv(file, parse_dates =["Date"]))#, usecols=daily_cols))
            
        daily_data = pd.concat(daily_data_list)
                                      
        data = intraday_data.merge(daily_data, how="left", left_on=["Date", "Id"], right_on=["Date", "ID"])
        
        data.index = data['Date'] + pd.to_timedelta(data['Time'])
        
        data.drop(["ID", "Date", "Time"], axis=1, inplace=True)
                                      
        return data.sort_index()
    
    def _intraday_file2date(self, file):
        return pd.to_datetime(file.stem)
    
    def _daily_file2date(self, file):
        return pd.to_datetime(file.stem[4:])
        
