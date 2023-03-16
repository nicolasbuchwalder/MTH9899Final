import numpy as np
import pandas as pd
import datetime as dt

class Preprocessor:
    """
    Preprocessing class that handles all the preprocessing for the project
    """
    def __init__(self, raw_df):
        self._raw_df = raw_df
        self._target = None
        self._daily_features = None
        self._intraday_features = None
        self._rolling_intraday_features = None
        self._rolling_daily_features = None
        
        
        
    def create_target(self, normalize_by_vol=True, clip_values=True, clip_quantiles=(0.01, 0.99)):
        """
        Function to create target each day from raw_data
        """
        # getting the cummulative return for each day and each id until 15:30 (included) and after 15:30 (excluded)
        # this allows to get in the "False" index (level 2) the cummulative return at 15:30 and in the "True" the one at 16:00
        day_sep = self._raw_df[["Id", "CumReturnResid"]].groupby(["Id", pd.Grouper(level=0, freq="D"), self._raw_df.index.time>dt.time(15,30)]).last()
        # let:
        #   return between yesterday 16:00 and today 15:30 be RB = (P(15:30 today) - P(16:00 yest.)) / P(16:00 yest.)
        #   return between yesterday 16:00 and today 16:30 be RT = (P(16:00 today) - P(16:00 yest.)) / P(16:00 yest.)
        #   return between today 15:30 and today 16:00 be RE = (P(16:00 today) - P(15:30 today)) / P(15:30 today)
        # we notice that RE = ((RB + 1) - (RT + 1)) / (RT + 1)
        # we compute the return of end of day as described above here:
        adj_end = day_sep.add(1).groupby(level=[0, 1]).pct_change().dropna().droplevel(2)
        # getting the cummulative return at 15:30 for every day and every id
        begin = day_sep.xs(False, level=2).shift(-1)
        # joining both together (this also removes the days that are incomplete
        joined = adj_end.merge(begin, how="inner", left_index=True, right_index=True)
        # let:
        #   return between yesterday 15:30 and yersterday 16:00 be RE = (P(16:00 yest.) - P(15:30 yest.)) / P(15:30 yest.)
        #   return between yesterday 16:00 and today 15:30 be RB = (P(15:30 today) - P(16:00 yest.)) / P(16:00 yest.)
        #   (our target) return between 15:30 yesterday and 15:30 today be T = (P(15:30 today) - P(15:30 yest.)) / P(15:30 yest.)
        # we notice that T = (CRE + 1) * (RE + 1) - 1
        # we compute the target as described above here:
        target = joined.stack().add(1).groupby(level=[0, 1]).prod().sub(1).rename("Target").to_frame()
        
        # normalizing data by daily estimated vol
        if normalize_by_vol:
            # daily estimated vol by Id and date
            est_vol = self._raw_df[["Id", "EST_VOL"]].groupby(["Id", pd.Grouper(level=0, freq="D")]).first()
            # regroup data together
            target_vol = target.merge(est_vol, how="left", left_index=True, right_index=True)
            # computed scaled by vol
            target = (target_vol["Target"] / target_vol["EST_VOL"]).rename("Target")
            
        # clipping values
        if clip_values:
            target.clip(lower=target.quantile(clip_quantiles[0]), upper=target.quantile(clip_quantiles[1]), inplace=True)
        
        self._target = target
        
        return self._target
    
    def create_daily_features(self):
        """
        Function to create features for each day
        """
        daily_returns = self._target.groupby(level=0).shift(1).dropna()
        
        return self._daily_features
        
    def create_raw_intraday_features(self):
        """
        Function to create features at each tick (remove cumulative)
        """
        # getting data by Id and datetime
        stacked_df = self._raw_df.set_index("Id", append=True).swaplevel(0,1)[["CumReturnResid", "CumVolume"]]
        # addind 1 to cumulative return (to calculate real return)
        stacked_df["CumReturnResid"] = stacked_df["CumReturnResid"].add(1)
        # grouping by id and day
        grouped_df = stacked_df.groupby([pd.Grouper(level=0), pd.Grouper(freq='1D', level=1)])
        tickdata = pd.DataFrame()
        # computing return: see "create_target" method to understand this way of computing returns
        tickdata["ResidReturn"] = grouped_df["CumReturnResid"].pct_change().fillna(stacked_df["CumReturnResid"].sub(1))
        # computing volume as the difference between each cumulative volume step
        tickdata["Volume"] = grouped_df["CumVolume"].diff().fillna(stacked_df["CumVolume"])
        
        self._intraday_features = tickdata.sort_index()
        
        return self._intraday_features
    
    def create_rolling_features(self, daily_ticks=20, intraday_ticks=26):
        
        self.create_raw_intraday_features()
        self._daily_features = self._target.groupby(level=0).shift(1).dropna()
        
        def strided_app(a, L):
            nrows = ((a.size-L))+1
            n = a.strides[0]
            return np.lib.stride_tricks.as_strided(a, shape=(nrows,L), strides=(n,n))
        
        intraday_data = []
        
        for Id, id_data in self._intraday_features.groupby(level=0):
            start_index = id_data.index.get_level_values(1)[0]
            filled_up_index = pd.date_range(start=start_index - pd.DateOffset(minutes=15*intraday_ticks), end=start_index, freq="15T", inclusive="left")
            filled_up_df = pd.DataFrame(np.nan, index=pd.MultiIndex.from_arrays([[Id] * len(filled_up_index), filled_up_index]), columns=id_data.columns)
            id_data = pd.concat([filled_up_df, id_data])
            arr = np.stack([strided_app(id_data["ResidReturn"].to_numpy(copy=False),L=intraday_ticks), strided_app(id_data["Volume"].to_numpy(copy=False),L=intraday_ticks)], axis=1)
            mask = id_data.index[intraday_ticks-1:].get_level_values(1).time == dt.time(15,30)
            days = id_data.index[intraday_ticks-1:].get_level_values(1)[mask].normalize()
            intraday_data.append(pd.DataFrame(arr[mask].reshape(-1, intraday_ticks * 2), index=pd.MultiIndex.from_arrays([[Id] * len(days), days])))
        rolling_intraday_features = pd.concat(intraday_data)
        rolling_intraday_features.columns = ["ResidReturnT-" + str(i) for i in range(rolling_intraday_features.shape[1]//2, 0, -1)] + ["Volume-" + str(i) for i in range(rolling_intraday_features.shape[1]//2, 0, -1)]
        
        self._rolling_intraday_features = rolling_intraday_features
        
        
        daily_data = []
        
        for Id, id_data in self._daily_features.groupby(level=0):
            start_index = id_data.index.get_level_values(1)[0]
            filled_up_index = pd.date_range(start=start_index - pd.DateOffset(days=daily_ticks), end=start_index, freq="D", inclusive="left")
            filled_up_df = pd.Series(np.nan, index=pd.MultiIndex.from_arrays([[Id] * len(filled_up_index), filled_up_index]), name=id_data.name)
            id_data = pd.concat([filled_up_df, id_data], axis=0)
            arr = strided_app(id_data.to_numpy(copy=False),L=daily_ticks)
            daily_data.append(pd.DataFrame(arr, index=id_data.index[daily_ticks-1:]))
        rolling_daily_features = pd.concat(daily_data)
        rolling_daily_features.columns = ["ResidReturnD-" + str(i) for i in range(rolling_daily_features.shape[1], 0, -1)] 
        
        self._rolling_daily_features = rolling_daily_features
        
        rolling_features = pd.concat([rolling_daily_features, rolling_intraday_features], axis=1)
        self._rolling_features = rolling_features.sort_index()
        return self._rolling_features
    
    def run(self, daily_ticks=20, intraday_ticks=26, scale_to_bps=True):
        self.create_target()
        self.create_rolling_features()
        self._target.index.rename(None, level=0, inplace=True)
        merged = self._rolling_features.merge(self._target, left_index=True, right_index=True)
        if scale_to_bps:
            return merged.mul(1e4)
        return merged
    
   
    
class Standardizer:
    
    def fit(self, series):
        exp_series = series.unstack(level=0)
        means = exp_series.mean()
        mean_means = means.mean()
        self._means_dict = means.to_dict()
        self._means_dict["_MEAN"] = mean_means
        stds = exp_series.std()
        mean_stds = stds.mean()
        self._stds_dict = stds.to_dict()
        self._stds_dict["_MEAN"] = mean_stds
        return self
    
    def transform(self, series):
        exp_series = series.unstack(level=0)
        def normalize(col):
            try:
                mean = self._means_dict[col.name]
                std = self._stds_dict[col.name] 
                if pd.isnull(std):
                    return col
                else:
                    return (col - mean) / std 
            except:
                return (col - self._means_dict["_MEAN"]) / self._stds_dict["_MEAN"]

        return exp_series.apply(normalize)
    
    def inverse_transform(self, series):
        pass
        

        
        