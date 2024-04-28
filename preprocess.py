import os
import pandas as pd

# Define input and output directories
input_dir = 'archive'
output_dir = 'preprocessed_archive'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Columns of interest
COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
SCOLS = ["vh","vl","vc","open_s", 'Adj Close_s', "volume_s"]
OBS_COLS = ['vh', 'vl', 'vc', 'open_s', 'Adj Close_s', 'volume_s', 'vh_roll_7', 'vh_roll_14', 'vh_roll_30', 'vl_roll_7', 'vl_roll_14', 'vl_roll_30', 'vc_roll_7', 'vc_roll_14', 'vc_roll_30', 'open_s_roll_7', 'open_s_roll_14', 'open_s_roll_30', 'Adj Close_s_roll_7', 'Adj Close_s_roll_14', 'Adj Close_s_roll_30', 'volume_s_roll_7', 'volume_s_roll_14', 'volume_s_roll_30']
EPISODE_LENGTH = 500

# Preprocess each CSV file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.csv'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Read CSV file into a DataFrame
        df = pd.read_csv(input_path)

        # Select columns of interest
        df = df[COLS]

        # Get number of rows dropped before preprocessing
        rows_dropped_before = len(df) - len(df.dropna())

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Convert 'Date' column to Timestamp format
        df['Date'] = pd.to_datetime(df['Date'])
        df["vh"] = df["High"]/df["Open"]
        df["vl"] = df["Low"]/df["Open"]
        df["vc"] = df["Close"]/df["Open"]
        df["open_s"] = df["Open"] - df["Open"].shift(1)
        df["Adj Close_s"] = df["Adj Close"] - df["Adj Close"].shift(1)
        df["volume_s"] = df["Volume"] - df["Volume"].shift(1)


        new_cols = []

        for col in SCOLS:
            df[col+"_roll_7"] = df[col].rolling(7).mean().bfill()
            new_cols.append(col+"_roll_7")
            df[col+"_roll_14"] = df[col].rolling(14).mean().bfill()
            new_cols.append(col+"_roll_14")
            df[col+"_roll_30"] = df[col].rolling(30).mean().bfill()
            new_cols.append(col+"_roll_30")

        # Save preprocessed DataFrame to a new CSV file
        df.to_csv(output_path, index=False)

         # Get number of rows dropped after preprocessing
        rows_dropped_after = rows_dropped_before - (len(df) - len(df.dropna()))

        print(f"Preprocessed {filename} and saved to {output_path}")
        print(f"Rows dropped: {rows_dropped_after}")
