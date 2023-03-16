import sys
import argparse
import datetime as dt
from pathlib import Path

class Parser:
    """
    Class that handle the arguments and transforms then into the right format
    """
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument('-m', required=True)
        self._parser.add_argument('-i', required=True)
        self._parser.add_argument('-o', required=True)
        self._parser.add_argument('-s', required=True)
        self._parser.add_argument('-e', required=True)
        self._parser.add_argument('-p')
        self._args = self._parser.parse_args()
        if int(self._args.m) == 2 and self._args.p == None:
            raise ValueError("Argument -p must be set for mode 2")
    
    # gets arguments in right format
    def convert_args(self):
        # checking and converting into mode number
        try:
            mode = int(self._args.m)
            if mode not in [1, 2]:
                raise ValueError("mode should be '1' or '2'")
        except:
            raise ValueError("mode should be '1' or '2'")
        # checking and converting paths
        try:
            inp = (Path.cwd()/ self._args.i).resolve()
            if inp == None:
                raise ValueError("could not parse input path (needs to be absolute)")
        except Exception as e:
            raise ValueError("could not parse input path (needs to be absolute)")
        try:
            out = (Path.cwd()/ self._args.o).resolve()
            if out == None:
                raise ValueError("could not parse input path (needs to be absolute)")
        except Exception as e:
            raise ValueError("could not parse input path (needs to be absolute)")

        # checking and converting dates
        try:
            input_date = dt.datetime.strptime(self._args.s, "%Y-%m-%d")
        except:
            raise ValueError("start date should be of format YYYY-MM-DD")
        
        try:
            print(self._args.e)
            end_date = dt.datetime.strptime(self._args.e, "%Y-%m-%d")
        except:
            raise ValueError("end date should be of format YYYY-MM-DD")
        
        try:
            mod = (Path.cwd()/ self._args.p).resolve()
            if mod == None:
                raise ValueError("could not parse models path (needs to be absolute)")
        except Exception as e:
            raise ValueError("could not parse models path (needs to be absolute)")
            
        return mode, inp, out, input_date, end_date, mod
