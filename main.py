from pathlib import Path
from arguments import Parser
from datahandling import DataHandler
from preprocessing import Preprocessor
from predictions import Predictor

def main():
    parser = Parser()
    mode, inp, out, start_date, end_date, mod = parser.convert_args()

    d = DataHandler()
    if mode == 1:
        print("reading raw data...")
        raw_data = d.read_raw(inp, start_date, end_date)
        print("generating targets and features...(takes time)")
        p = Preprocessor(raw_data, mod)
        dataset = p.create_dataset()
        print("storing data...")
        d.store_dataset(out, dataset)

    else:
        print("reading daily features and targets...")
        X, y, weights = d.read_processed(inp, start_date, end_date)
        print("loading model...")
        p = Predictor(mod)
        print("predicting...")
        y_preds = p.predict(X)
        p.evaluate(y_preds, y, weights)
        print("storing predictions...")
        d.store_predictions(out, y_preds, y)

if __name__ == "__main__":
    main()
