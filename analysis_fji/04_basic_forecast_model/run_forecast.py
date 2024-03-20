import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor
from pathlib import Path
import os
from datetime import datetime

from utils import get_combined_dataset, get_municipality_grids
from input_dataset import create_input_dataset, create_windfield_dataset, create_rainfall_dataset, trigger_fji
from predict_damage import apply_model
import zipfile

def main():
    # Get today's date in YYMMDD format
    today_date = datetime.now().strftime("%Y%m%d")
    try:
        # Getting data from ECMWF
        df_windfield = create_windfield_dataset(thres=120, deg=3)

        # Check if some forecasts take place on Fiji
        trigger = trigger_fji(df_windfield=df_windfield)

        if trigger:
            # Get rainfall data (might take a couple of minutes- more like ~4 minutes)
            df_rainfall = create_rainfall_dataset(df_windfield)

            # Load rainfall data (only if there are some wind forecasts on Fiji)
            rain_dir = (
                Path(os.getenv("STORM_DATA_DIR"))
                / "analysis_fji/02_model_features/03_rainfall/output"
            )

            filename = rain_dir / "NOMADS"/ today_date /"rainfall_data_rw_mean.csv"
            df_rainfall = pd.read_csv(filename)

            # Merging windspeed data with stationary data
            df_input = create_input_dataset(df_windfield, df_rainfall)

            # Group the DataFrame by the 'unique_id' column
            grouped = df_input.groupby('unique_id')

            # Create a list of Forecasts DataFrames, one for each unique_id
            list_forecast = [group for name, group in grouped]

            # Apply model
            list_df_out = apply_model(list_forecast)

            # If you want to add more information to the output dataset (like the time period the windspeed was measured)
            list_output = []
            for i, df_out in enumerate(list_df_out):
                df_aux = df_out.copy()
                time_init, time_end = list_forecast[i][['time_init', 'time_end']].iloc[0]
                npoints = len(df_aux)

                df_aux['forecast_time_init'] = [time_init] * npoints
                df_aux['forecast_time_end'] = [time_end] * npoints
                list_output.append(df_aux)
                # Create artifact folder with CSV files
                artifact_folder = Path(f'output_{today_date}')
                artifact_folder.mkdir(parents=True, exist_ok=True)
                # Save each generated DataFrame as a CSV file within 'outputs_YYMMDD'
                for i, df_out in enumerate(list_output):
                    df_out.to_csv(artifact_folder /f'output_{i}.csv', index=False)

                # Zip the folder
                # with zipfile.ZipFile(f'output_{today_date}.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
                #     for root, _, files in os.walk(artifact_folder):
                #         for file in files:
                #             zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), artifact_folder))

        else:
            # Create a TXT file with a message
            with open(f'output_{today_date}.txt', 'w') as txt_file:
                txt_file.write(f'Trigger condition not met. No output generated.\nDate={today_date}')
    except:
        # Create a TXT file with a message
            with open(f'output_{today_date}.txt', 'w') as txt_file:
                txt_file.write(f'ECMWF not responding. Try again later.\nDate={today_date}')


if __name__ == "__main__":
    main()
