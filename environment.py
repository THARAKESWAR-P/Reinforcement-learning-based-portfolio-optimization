import os

import abc
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn import preprocessing

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import config
import warnings
warnings.filterwarnings("ignore")


tf.compat.v1.enable_v2_behavior()

class PortfolioEnv(py_environment.PyEnvironment):

  def __init__(self):
    # for stock in config.COINS:
    #    for col in config.
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(len(config.COINS)+1,), dtype=np.float64, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(len(config.OBS_COLS),), dtype=np.float64, minimum=config.OBS_COLS_MIN, maximum=config.OBS_COLS_MAX, name='observation')
    # self._state = 0
    self.reset()
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._episode_ended = False
    self.index = 0
    self.time_delta = pd.Timedelta(days=1)
    self.memory_return = pd.DataFrame(columns=[t+"_close" for t in config.COINS])
    self.init_cash = config.INIT_CASH
    self.current_cash = self.init_cash
    self.current_value = self.init_cash
    self.previous_price = {}
    self.old_dict_coin_price_1 = {}
    self.old_dict_coin_price_2 = {}

    self.money_split_ratio = np.zeros((len(config.COINS)+1))
    self.money_split_ratio[0] = 1

    self.common_dates = set()

    # Reset DataFrame for portfolio data
    self.portfolio_df = pd.DataFrame(columns=config.COLS + ['Stock'])
    for stock_file in config.STOCK_FILES:
        stock_df = pd.read_csv(os.path.join(config.DATA_DIR, stock_file))
        # Convert 'Date' column to Timestamp
        stock_df["Date"] = pd.to_datetime(stock_df["Date"])
        # Sort DataFrame by 'Date'
        stock_df = stock_df.sort_values("Date")

        if len(self.common_dates) == 0 :
           self.common_dates = set(stock_df["Date"])
        else:
           self.common_dates = self.common_dates.intersection(set(stock_df["Date"]))

        # Reset index
        stock_df = stock_df.reset_index(drop=True)
        stock_df = stock_df.ffill().bfill()
        # Add 'Stock' column to indicate the stock name
        stock_df['Stock'] = os.path.splitext(stock_file)[0]  # Extract stock name from file name
        # Append stock data to portfolio DataFrame
        self.portfolio_df = pd.concat([self.portfolio_df, stock_df], ignore_index=True)

    # Initialize StandardScaler
    self.scaler = preprocessing.StandardScaler()
    
    # Preprocess DataFrame
    self.portfolio_df = self.portfolio_df[self.portfolio_df["Stock"].isin(config.COINS)].sort_values("Date")
    self.scaler.fit(self.portfolio_df[config.SCOLS].values)
    self.portfolio_df = self.portfolio_df.reset_index(drop=True)
    

    # Initialize time window
    # self.max_index = self.portfolio_df.shape[0]
    # # start_point = np.random.choice(np.arange(3, self.max_index - config.EPISODE_LENGTH))
    # start_point = 0
    # end_point = start_point + config.EPISODE_LENGTH
    # self.portfolio_df = self.portfolio_df.loc[start_point:end_point+2].reset_index(drop=True)

    self.max_index = len(self.common_dates)
    # print('No. Dates: {0}'.format(self.max_index))
    start_point = np.random.choice(np.arange(len(config.COINS), self.max_index - config.EPISODE_LENGTH))
    end_point = start_point + config.EPISODE_LENGTH
    self.current_index = start_point
    
    # Set initial time and create initial DataFrame slice
    self.common_dates = sorted(self.common_dates)
    print(self.common_dates[0])
    print(self.common_dates[-1])
    self.init_date = self.common_dates[start_point]
    self.current_date = self.init_date
    self.dfslice = self.portfolio_df[self.portfolio_df["Date"] == self.current_date].copy()

    self.current_stock_num_distribution = self.calculate_actual_shares_from_money_split()
    self.previous_value = self.current_value
    self.current_stock_money_distribution,self.current_value  = self.calculate_money_from_num_stocks()
    self.money_split_ratio = self.normalize_money_dist()

    self.step_reward = 0
    info_ =  {
               "state":"state",\
               "money_split":self.money_split_ratio,\
               "share_num":self.current_stock_num_distribution,\
               "value":self.current_value,\
               "date":self.current_date,\
               "reward":self.step_reward,\
               "raw_output":self.get_observations_unscaled(),
               "scaled_output":self.get_observations()
             }
    self._state = info_["scaled_output"][config.OBS_COLS].values.flatten()
    reward = info_["reward"] / config.INIT_CASH
    self._episode_ended = True if self.index==config.EPISODE_LENGTH//3 else False
    
    return ts.restart(self._state)


  def _step(self, action):
    if self._episode_ended:
        return self.reset()
    
    if sum(action)<=1e-3:
        # self.money_split_ratio = np.zeros((len(config.COINS)+1))
        # self.money_split_ratio[0] = 1
        self.money_split_ratio = [1/len(action) for t in action]
    else:
        self.money_split_ratio = [t/sum(action) for t in action]

    self.current_stock_num_distribution = self.calculate_actual_shares_from_money_split()
    self.step_time()
    self.index += 1

    info_ = {
              "state":"state",\
              "money_split":self.money_split_ratio,\
              "share_num":self.current_stock_num_distribution,\
              "value":self.current_value,\
              "date":self.current_date,\
              "reward":self.step_reward,\
              "scaled_output":self.get_observations()
            }

    reward = info_["reward"]
    self._state = info_["scaled_output"][config.OBS_COLS].values.flatten()
    # print('date: {2}, action: {0}\n state: {1}\n reward: {3}\n msr: {4}\n csmd: {5}'.format(action, self._state, self.current_date, reward, self.money_split_ratio, self.current_stock_money_distribution))

    self._episode_ended = True if self.index==config.EPISODE_LENGTH//3 or self.current_cash < 1 else False
    if self._episode_ended:
        reward = 0
        return ts.termination(self._state , reward)
    else:
        try:
            return ts.transition(
                self._state, reward=reward, discount=1)
        except Exception as e:
            print("ERRORRRRRR!!!!!!!!!!!!!!!!")
            print(self._state)
            print(reward)
            print(self.step_reward, self.current_value, self.previous_value)
            print(self.current_stock_money_distribution)
            print(self.current_stock_num_distribution)
            print(action)
            print(self.index)
            print(self.dfslice)
            print(self.current_date)
            print(self.money_split_ratio )
            print(e)
            self.portfolio_df.to_csv(os.path.join(config.LOGDIR,"error_df.csv"))
            
            raise ValueError
  
  def step_time(self):
    self.current_index = self.current_index + 1
    self.current_date = self.common_dates[self.current_index]
    self.dfslice = self.portfolio_df[self.portfolio_df["Date"] == self.current_date].copy()
    self.previous_value = self.current_value
    self.current_stock_money_distribution, self.current_value = self.calculate_money_from_num_stocks()
    self.money_split_ratio = self.normalize_money_dist()
    self.step_reward = self.current_value - self.previous_value


  def get_observations(self):
    dfslice = self.dfslice
    dfs = pd.DataFrame()
       
    for i,grp in dfslice.groupby("Stock"):
        tempdf = pd.DataFrame(self.scaler.transform(grp[config.SCOLS].values))
        tempdf.columns = [i+"_"+c for c in config.SCOLS]
        if dfs.empty:
            dfs = tempdf
        else:
            dfs = dfs.merge(tempdf,right_index=True,left_index=True,how='inner')

    return dfs


  def get_observations_unscaled(self):
    dfslice = self.dfslice
    dfs = pd.DataFrame()
    for i,grp in dfslice.groupby("Stock"):
        tempdf = pd.DataFrame(grp[config.COLS].values)
        tempdf.columns = [i+"_"+c for c in config.COLS]
        if dfs.empty:
            dfs = tempdf
        else:
            dfs = dfs.merge(tempdf,right_index=True,left_index=True,how='inner')
    
    self.memory_return = pd.concat([self.memory_return,dfs[[t+"_Close" for t in config.COINS]]],ignore_index=True)
    
    return dfs


  def calculate_actual_shares_from_money_split(self):
    dict_stock_price = self.dfslice[["Stock", "Open"]] \
                        .set_index("Stock").to_dict()["Open"]
    num_shares = []
    for i, stock in enumerate(config.COINS):
        stock_price = dict_stock_price.get(stock, self.old_dict_coin_price_1.get(stock, None))
        if stock_price is not None:
            num_shares.append(self.money_split_ratio[i + 1] * self.current_value // stock_price)
        else:
            # If current stock price is not available, use the previous price
            num_shares.append(self.money_split_ratio[i + 1] * self.current_value // self.old_dict_coin_price_1[stock])

    # Update current cash and save current stock prices for next iteration
    self.current_cash = self.money_split_ratio[0] * self.current_value
    for stock, price in dict_stock_price.items():
        self.old_dict_coin_price_1[stock] = price

    return num_shares


  def calculate_money_from_num_stocks(self):
    money_dist = []
    money_dist.append(self.current_cash)
    dict_coin_price = self.dfslice[["Stock","Open"]]\
                    .set_index("Stock").to_dict()["Open"]
    for i,c in enumerate(config.COINS):
        if c in dict_coin_price:
            money_dist.append(self.current_stock_num_distribution[i]*dict_coin_price[c])
        else:
            money_dist.append(self.current_stock_num_distribution[i]*self.old_dict_coin_price_2[c])
    
    for c in dict_coin_price:
        self.old_dict_coin_price_2[c] = dict_coin_price[c]

    return money_dist,sum(money_dist)


  def normalize_money_dist(self):
    normal = []
    for i,c in enumerate(self.current_stock_money_distribution):
        normal.append(c/self.current_value)

    return normal   


# if __name__ == '__main__':
#   environment = PortfolioEnv()
#   time_step = environment.current_time_step()
#   next_time_step = environment._step([100, 318, 0, 500])
#   info_ = {
#               "state":"state",\
#               "money_split":environment.money_split_ratio,\
#               "share_num":environment.current_stock_num_distribution,\
#               "value":environment.current_value,\
#               "date":environment.current_date,\
#               "reward":environment.step_reward,\
#               "scaled_output":environment.get_observations()
#             }

#   state = info_["scaled_output"][config.OBS_COLS].values.flatten()
#   print(state)