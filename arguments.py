import sys
import argparse

class Parser:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument('-m', required=True)
        
