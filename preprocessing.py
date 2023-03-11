import pandas as pd
import datetime as dt

class Preprocessor:
    """
    Preprocessing class that handles all the preprocessing for the project
    """
    def create_target(self, df):
        # getting the cummulative return for each day and each id until 15:30 (included) and after 15:30 (excluded)
        # this allows to get in the "False" index (level 2) the cummulative return at 15:30 and in the "True" the one at 16:00
        day_sep = df[["Id", "CumReturnResid"]].groupby(["Id", df.index.date, df.index.time>dt.time(15,30)]).last()
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
        target = joined.stack().add(1).groupby(level=[0, 1]).prod().sub(1).to_frame()
        return target
    def oui(self):
        pass