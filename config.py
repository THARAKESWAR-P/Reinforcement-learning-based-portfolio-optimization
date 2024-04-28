import os
import pandas as pd
import tensorflow as tf

DATA_DIR = 'preprocessed_archive'
STOCK_FILES = [ 'ALX.csv', 'CSL.csv', 'DOW.csv', 'FLT.csv', 'ILU.csv', 'NCM.csv']
FILE = "cleaned_preprocessed.csv"
COINS = [ 'ALX', 'CSL', 'DOW', 'FLT', 'ILU', 'NCM']
COLS = ['Open','High','Low','Close', 'Adj Close', 'Volume']
TCOLS = ['vh', 'vl', 'vc', 'open_s', 'volume_s', 'vh_roll_30', 'vl_roll_30', 'vc_roll_30', 'open_s_roll_30', 'volume_s_roll_30']
# COLS = ['high', 'low', 'open', 'close', 'volume', 'quoteVolume','weightedAverage']
SCOLS = ['vh', 'vl', 'vc', 'open_s', 'volume_s', 'vh_roll_7', \
    'vh_roll_14', 'vh_roll_30', 'vl_roll_7', 'vl_roll_14', 'vl_roll_30', 'vc_roll_7', 'vc_roll_14', 'vc_roll_30', \
        'open_s_roll_7', 'open_s_roll_14', 'open_s_roll_30', 'volume_s_roll_7', 'volume_s_roll_14', 'volume_s_roll_30']

OBS_COLS = ['ALQ_vh','AGL_vh','ALL_vh','ALQ_vl','AGL_vl','ALL_vl','ALQ_vc','AGL_vc','ALL_vc','ALQ_open_s','AGL_open_s','ALL_open_s','ALQ_volume_s','AGL_volume_s','ALL_volume_s','ALQ_vh_roll_30','AGL_vh_roll_30','ALL_vh_roll_30','ALQ_vl_roll_30','AGL_vl_roll_30','ALL_vl_roll_30','ALQ_vc_roll_30','AGL_vc_roll_30','ALL_vc_roll_30','ALQ_open_s_roll_30','AGL_open_s_roll_30','ALL_open_s_roll_30','ALQ_volume_s_roll_30','AGL_volume_s_roll_30','ALL_volume_s_roll_30']                                                                  
OBS_COLS_MIN = [-0.39,-0.39,-0.39,-52.92,-144.06,-159.81,-31.78,-19.57,-61.76,-87.7,-65.71,-130.35,-51.36,-51.09,-77.06,-0.77,-0.77,-0.77,-12.17,-14.24,-25.47,-15.64,-12.37,-12.73,-44.75,-25.01,-32.13,-39.2,-83.44,-74.04]
OBS_COLS_MAX = [53.16,69.71,90.97,0.4,0.4,0.4,39.31,24.4,85.49,114.25,66.32,130.35,55.72,60.25,66.09,9.96,17.79,53.89,0.81,0.81,0.81,7.58,9.9,14.93,41.44,16.35,32.11,34.64,82.48,59.62]

# Function to update OBS_COLS, OBS_COLS_MIN, and OBS_COLS_MAX based on selected COINS
def update_observation_columns():
    # Initialize observation columns list and min/max dictionaries
    observation_columns = []
    obs_cols_min = []
    obs_cols_max = []

    # Iterate over selected COINS to create observation columns list and min/max dictionaries
    for coin in COINS:
        # Read data for the current coin from the corresponding file
        coin_file = f"{coin}.csv"
        coin_df = pd.read_csv(os.path.join(DATA_DIR, coin_file))
        
        # Create columns related to the current coin and update min/max dictionaries
        coin_cols = [f"{coin}_{col}" for col in TCOLS]  # Assuming your columns have prefix with coin name
        observation_columns.extend(coin_cols)
        obs_cols_min.extend(coin_df[TCOLS].min().tolist())
        obs_cols_max.extend(coin_df[TCOLS].max().tolist())

    # Update OBS_COLS, OBS_COLS_MIN, and OBS_COLS_MAX
    globals()['OBS_COLS'] = observation_columns
    globals()['OBS_COLS_MIN'] = obs_cols_min
    globals()['OBS_COLS_MAX'] = obs_cols_max

# Initial call to update observation columns based on the initial COINS
update_observation_columns()

EPISODE_LENGTH = 1500
LOGDIR="LOGDIR"
MODEL_SAVE = "model_save"

INIT_CASH = 30000
NUM_ITERATIONS = 1000
COLLECT_STEPS_PER_ITERATION = 150
LOG_INTERVAL = 10
EVAL_INTERVAL = 4
MODEL_SAVE_FREQ = 12


REPLAY_BUFFER_MAX_LENGTH = 10000 #100000
BATCH_SIZE = 100
NUM_EVAL_EPISODES = 4

actor_fc_layers=(400, 300)
critic_obs_fc_layers=(400,)
critic_action_fc_layers=None
critic_joint_fc_layers=(300,)
ou_stddev=0.2
ou_damping=0.15
target_update_tau=0.05
target_update_period=5
dqda_clipping=None
td_errors_loss_fn=tf.compat.v1.losses.huber_loss
gamma=0.05
reward_scale_factor=1
gradient_clipping=None

actor_learning_rate=1e-4
critic_learning_rate=1e-4
debug_summaries=False
summarize_grads_and_vars=False