import time
import pandas as pd
from datetime import datetime


LOG_FILE = "monitoring_log.csv"


def log_prediction(prediction):
    data = {
        "timestamp": [datetime.now()],
        "prediction": [prediction]
    }

    df = pd.DataFrame(data)

    try:
        old_data = pd.read_csv(LOG_FILE)
        df = pd.concat([old_data, df], ignore_index=True)
    except FileNotFoundError:
        pass

    df.to_csv(LOG_FILE, index=False)


def show_monitoring():
    try:
        df = pd.read_csv(LOG_FILE)

        print("\n==============================")
        print("MODEL MONITORING")
        print("==============================")

        print("Total Predictions:", len(df))

        print("\nPrediction Distribution:")
        print(df["prediction"].value_counts())

        print("\nLatest Predictions:")
        print(df.tail())

    except FileNotFoundError:
        print("No prediction data available yet.")


if __name__ == "__main__":

    print("Monitoring system started...")

    while True:
        show_monitoring()
        time.sleep(30)