import csv
import os
import random
import pandas as pd
import gym
import stable_baselines3
from gym import spaces
import numpy as np
from plotly.io._orca import psutil

import shimmy
import ta
import plotly.graph_objects as go
from collections import deque
import psutil
import os
import tracemalloc
from memory_profiler import profile
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback
import optuna
from stable_baselines3 import PPO, DQN
from sb3_contrib import QRDQN, TRPO
import optuna
from stable_baselines3.common.monitor import Monitor

"""
할 일

self.observation이 step이 지나감에 따라 같이 update시키기. -- 완료.
    - self.observation의 크기를 유지 하기 위해 window_siz는 유지하며 step을 따라가도록 변경 --완료
    - current_step이 df보다 커지거나 같아질 경우 done = True로 바꾸도록 변경  -- 완료

render 함수 만들기.(가시화)

"""


class stablebaselineEnv(gym.Env):
    # @profile
    def __init__(self, df, full_window_size, test_window_size, usdt_balance=1000, btc_size=0, leverage=10):
        super(stablebaselineEnv, self).__init__()
        self.df = df
        self.full_window_size = full_window_size
        self.test_window_size = test_window_size
        self.window_size = full_window_size - test_window_size
        self.df['Volume'] = np.log(self.df['Volume'] + 1e-12) / 10
        self.slice_deque = deque(maxlen=self.full_window_size)
        self.obs_deque = deque(maxlen=self.window_size)
        self.train_deque = deque(maxlen=self.test_window_size)
        self.slice_df, self.obs_df, self.train_df = self.generate_random_data_slice(df, full_window_size, test_window_size)  # 랜덤 위치로 slice된 차트 데이터 초기화

        self.mean = np.mean(self.obs_df, axis=0)  # 0번 모드 정규화를 할 때 필요함
        self.std = np.std(self.obs_df, axis=0)  # 0번 모드 정규화를 할 때 필요함
        self.standard_slice_df, self.standard_obs_df, self.standard_train_df = stablebaselineEnv.standardzation(self, self.slice_df, self.obs_df, self.train_df, self.mean, self.std, mode=1)  # 정규화를 진행함. 모드 0,1 있음.

        self.final_slice_df, self.final_obs_df, self.final_train_df = stablebaselineEnv.add_indicator(self, self.standard_slice_df)

        self.action_space = spaces.Discrete(4)  # 0: Long, 1: Short, 2: Close, 3: Hold
        self.observation_space = spaces.Dict({
            "chart_data": spaces.Box(low=0, high=len(self.final_obs_df), shape=self.final_obs_df.shape, dtype=np.float32),  # 차트 데이터
            "position": spaces.Discrete(3),  # 포지션 {0:Long, 1:Short, 2:None}
            "action": spaces.Discrete(4),  # 액션 {0:Long, 1:Short, 2:Close, 3:Hold}
            "current_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 현재 가격
            "avg_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 평균 진입 가격
            "pnl": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),  # 미실현 손익
            "total_pnl": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),  # 누적 손익
            # "usdt_balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # USDT 잔고
            "size": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 포지션 수량
            "margin": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 사용 중인 마진
            "total_balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)  # 총 자산
        })

        self.full_window_size = full_window_size
        self.test_window_size = test_window_size

        self.start_step = self.full_window_size - self.test_window_size
        self.current_step = self.final_obs_df.tail(1)
        self.current_price = round(random.uniform(self.final_train_df['Open'].iloc[0], self.final_train_df['Close'].iloc[0]), 2)  # 현재 가격을 시가, 종가 사이 랜덤 값으로 결정

        # reset 미포함
        self.initial_usdt_balance = usdt_balance  # 초기 usdt 잔고
        self.min_order = 0.002  # 최소 주문 수량
        self.fee = 0.0005  # 거래 수수료
        self.leverage = leverage  # 레버리지

        # reset 포함
        self.usdt_balance = usdt_balance  # 초기 usdt 잔고
        self.btc_size = btc_size  # 포지션 수량
        self.margin = 0  # 포지션 증거금
        self.position = 2  # 포지션 {0:Long, 1:Short, 2:None}
        self.order_price = 0  # 주문 금액
        self.last_size_value = 0  # (평단가 계산에 필요)
        self.current_avg_price = 0  # 현재 평단가
        self.pnl = 0  # 미실현 손익
        self.closing_pnl = 0  # 실현 손익
        self.total_pnl = 0  # 누적 손익
        self.total_fee = 0  # 누적 수수료
        self.total_balance = 0  # 총 자산
        self.action_history = pd.DataFrame(columns=['action'])

        self.current_index = self.slice_df.index.get_loc(self.current_step.index[0]) + 1

    def reset(self):  # 리셋 함수
        self.slice_df, self.obs_df, self.train_df = self.generate_random_data_slice(self.df, self.full_window_size, self.test_window_size)  # 랜덤 위치로 slice된 차트 데이터 초기화
        self.mean = np.mean(self.obs_df, axis=0)
        self.std = np.std(self.obs_df, axis=0)
        self.standard_slice_df, self.standard_obs_df, self.standard_train_df = stablebaselineEnv.standardzation(self, self.slice_df, self.obs_df, self.train_df, self.mean, self.std, mode=1)  # 정규화를 진행함. 모드 0,1 있음.
        self.final_slice_df, self.final_obs_df, self.final_train_df = stablebaselineEnv.add_indicator(self, self.standard_slice_df)

        self.start_step = self.full_window_size - self.test_window_size
        self.current_step = self.obs_df.tail(1)
        self.current_price = round(random.uniform(self.final_train_df.iloc[0]['Open'], self.final_train_df.iloc[0]['Close']), 2)  # 현재 가격을 시가, 종가 사이 랜덤 값으로 결정

        self.usdt_balance = self.initial_usdt_balance  # 초기 usdt 잔고
        self.btc_size = 0  # 포지션 수량
        self.margin = 0  # 포지션 증거금
        self.position = 2  # 포지션 {0:Long, 1:Short, 2:None}
        self.order_price = 0  # 주문 금액
        self.last_size_value = 0  # (평단가 계산에 필요)
        self.current_avg_price = 0  # 현재 평단가
        self.pnl = 0  # 미실현 손익
        self.closing_pnl = 0  # 실현 손익
        self.total_pnl = 0  # 누적 손익
        self.total_fee = 0  # 누적 수수료
        self.total_balance = 0  # 총 자산
        self.action_history = pd.DataFrame(columns=['action'])
        self.current_index = self.final_slice_df.index.get_loc(self.current_step.index[0]) + 1
        return self.get_obs()

    def seed(self, seed=42):
        np.random.seed(seed)
        return [seed]

    # 나중에 수량지정을 위한 함수 (min_order부분만 바꾸면됌)

    def cac_order_size(self):
        order_size = self.min_order
        return order_size

    # action을 수행할 수 있는 최소한의 조건 확인

    def act_check(self, action):
        required_margin = (self.cac_order_size() * self.current_price) / self.leverage

        if action == 0 or action == 1:
            if self.position == action or self.position == 2 or self.position is None:
                if self.usdt_balance > required_margin:
                    return action
                else:
                    return 3
            else:
                if self.usdt_balance + self.margin + self.pnl > required_margin:
                    return action
                else:
                    return 3

        elif action == 2:
            if self.position == 0 or self.position == 1:
                return 2
            else:
                return 3

        elif action == 3 or self.position is None:
            return 3

    # 포지션 진입

    def open_position(self, action):
        order_size = self.cac_order_size()
        required_margin = (order_size * self.current_price) / self.leverage
        open_fee = order_size * self.current_price * self.fee

        self.usdt_balance -= required_margin + open_fee
        self.btc_size += order_size
        self.margin += required_margin

        self.order_price = order_size * self.current_price
        self.current_avg_price = (self.order_price + self.last_size_value) / self.btc_size
        self.last_size_value = self.btc_size * self.current_price

        self.pnl = (1 if action == 0 else -1) * (
                self.current_price - self.current_avg_price) * self.btc_size * self.leverage

        self.total_fee -= open_fee
        self.total_balance = self.usdt_balance + self.margin
        self.position = action

    def close_position(self):
        closing_fee = self.btc_size * self.current_price * self.fee
        closing_pnl = (1 if self.position == 0 else -1) * (
                self.current_price - self.current_avg_price) * self.btc_size * self.leverage

        self.usdt_balance += self.margin + closing_pnl - closing_fee
        self.total_fee -= closing_fee
        self.total_pnl += closing_pnl
        self.closing_pnl = closing_pnl

        self.btc_size = 0
        self.margin = 0
        self.pnl = 0
        self.last_size_value = 0

        self.total_balance = self.usdt_balance + self.margin
        self.position = 2

    def act(self, action):
        action = self.act_check(action)
        if action == 0 or action == 1:  # Long or Short
            if self.position == action or self.position == 2 or self.position is None:
                self.open_position(action)
            else:
                self.close_position()
                self.open_position(action)

        elif action == 2:  # Close
            if self.position == 0 or self.position == 1:
                self.close_position()


        elif action == 3:  # Hold
            if self.position == 0:  # Long
                self.pnl = (self.current_price - self.current_avg_price) * self.btc_size * self.leverage
            elif self.position == 1:  # Short
                self.pnl = (self.current_avg_price - self.current_price) * self.btc_size * self.leverage

            self.total_balance = self.usdt_balance + self.margin
        return action

    '''
    액션 :
        action : 0=Long, 1=Short, 2=Close, 3=Hold
        position : 0=Long, 1=Short, 2=None
        ex) (0.002*68000)/1=136, (0.002*68000)/2=68 필요 증거금 계산 예시 #

        return : self.position, self.acutal_action, self.pnl, self.closing_pnl, self.total_pnl, self.total_balance

        주문 수량은 일단 항상 최소 주문 금액으로 하겠습니다.
        최소 수량으로 해도 0.002개 이고 1배율일 경우 증거금 136usdt 정도 들어갑니다.

        추후 수량이 커질시 미결손실 또한 고려해야함
    '''

    # df 데이터를 받아 full_window_size만큼 랜덤 위치로 잘라서 Obs_df와 train_df로 나눈 df를 반환해주는 함수

    def generate_random_data_slice(self, df, full_window_size, test_window_size):
        strat_index = np.random.randint(0, len(df) - full_window_size)
        end_index = strat_index + full_window_size
        obs_end = end_index - test_window_size

        slice_df = df[strat_index:end_index]
        obs_df = df[strat_index:obs_end]
        train_df = df[obs_end:end_index]
        self.slice_deque.clear()
        self.obs_deque.clear()
        self.train_deque.clear()

        self.slice_deque.extend(slice_df.to_dict('records'))
        self.obs_deque.extend(obs_df.to_dict('records'))
        self.train_deque.extend(train_df.to_dict('records'))

        # return pd.DataFrame(self.obs_deque), pd.DataFrame(self.train_deque)
        return pd.DataFrame(self.slice_deque), pd.DataFrame(self.obs_deque), pd.DataFrame(self.train_deque)

    # 다음 step과 가격을 가져옴

    def next_obs(self):
        row_to_move = self.final_train_df.iloc[0:1]
        self.final_obs_df = pd.concat([self.final_obs_df.iloc[1:], row_to_move])
        self.final_train_df = self.final_train_df.iloc[1:]
        self.current_step = self.final_obs_df.tail(1)

    def get_price(self):
        open = self.final_train_df.iloc[0]['Open']
        close = self.final_train_df.iloc[0]['Close']
        self.current_price = round(random.uniform(open, close), 2)  # 현재 가격을 시가, 종가 사이 랜덤 값으로 결정

    # def calculate_reward(self):
    #     reward = 0
    #     if self.btc_size > 0 and self.current_avg_price > 0:
    #         reward = self.closing_pnl / (self.btc_size * self.current_avg_price)
    #     else:
    #         reward = 0
    #     self.closing_pnl = 0
    #     return reward

    def calculate_reward(self):
        # 현재 포지션이 있는 경우에만 리워드 계산
        reward = self.pnl
        return reward

    def get_obs(self, action=3):
        obs = {
            "chart_data": self.final_obs_df.values,
            "position": self.position,
            "action": action,
            "current_price": np.array([self.current_price]),
            "avg_price": np.array([self.current_avg_price]),
            "pnl": np.array([self.pnl]),
            "total_pnl": np.array([self.total_pnl]),
            # "usdt_balance": np.array([self.usdt_balance]),
            "size": np.array([self.btc_size]),
            "margin": np.array([self.margin]),
            "total_balance": np.array([self.total_balance / self.initial_usdt_balance])
        }
        return obs

    def step(self, action):
        self.next_obs()  # 다음 obs를 가져옴
        self.get_price()  # 현재 가격을 가져옴
        action = self.act(action)  # action을 수행함.
        reward = self.calculate_reward()  # reward 계산
        action_row = pd.DataFrame({'action': [action]}, index=[self.final_slice_df.index[self.current_index]])
        self.action_history = pd.concat([self.action_history, action_row])  # action_history에 action 추가
        self.current_index += 1
        if len(self.final_train_df) == 1:
            done = True
        else:
            done = False
        info = {'total_pnl': self.total_pnl}  # info 딕셔너리에 'total_pnl' 키와 값을 추가
        # if self.total_balance < self.total_balance * 0.3:
        #     done = True
        # else:
        #     done = False

        # gc.collect() #메모리 정리 코드
        # print(reward)
        return self.get_obs(action), reward, done, info

    def add_indicator(self, df):
        # RSI & ADX 보조지표 추가
        df['RSI_7'] = ta.momentum.rsi(df['Open'], window=7) / 100  # 노멀라이즈를 위해 100으로 나눠줌
        df['RSI_14'] = ta.momentum.rsi(df['Open'], window=14) / 100
        df['RSI_21'] = ta.momentum.rsi(df['Open'], window=21) / 100

        # # 단순 이동평균선 추가
        df['SMA_10'] = ta.trend.sma_indicator(df['Open'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Open'], window=20)
        df['SMA_60'] = ta.trend.sma_indicator(df['Open'], window=60)

        # 볼린저 밴드 추가
        bollinger = ta.volatility.BollingerBands(close=df['Open'], window=20, window_dev=2)
        df['Bollinger_High_20'] = bollinger.bollinger_hband()
        # df['Bollinger_Mid'] = bollinger.bollinger_mavg() #window가 20일 때 20일선과 같은것으로 알고 있음
        df['Bollinger_Low_20'] = bollinger.bollinger_lband()

        # 볼린저 밴드 추가
        bollinger = ta.volatility.BollingerBands(close=df['Open'], window=14, window_dev=2)
        df['Bollinger_High_14'] = bollinger.bollinger_hband()
        # df['Bollinger_Mid'] = bollinger.bollinger_mavg() #window가 14일 때 14일선과 같은것으로 알고 있음
        df['Bollinger_Low_14'] = bollinger.bollinger_lband()

        df = df.fillna(0)
        final_slice_df = df[0:self.full_window_size]
        final_obs_df = df[0:self.window_size]
        final_train_df = df[self.window_size:self.full_window_size]

        return final_slice_df, final_obs_df, final_train_df

    def standardzation(self, slice_df, obs_df, train_df, mean, std, mode=0):
        if mode == 0:  # 기존 데이터의 평균 및 표준편차 값으로 새로운 데이터 모두
            standard_slice_df = (slice_df - mean) / std
            standard_obs_df = (obs_df - mean) / std
            standard_train_df = (train_df - mean) / std


        # 이거 나눠서 정규화 해버리면 nan0떄문에 이상해짐
        elif mode == 1:  # 인터넷에 있는 방법, 바로 전 종가 값으로 현재 모든 값을 나눔. (볼륨 제외)
            scale = 1000
            standard_slice_df = pd.DataFrame()
            standard_slice_df_volume = slice_df['Volume']

            # 스케일링 대상 컬럼 리스트
            for column in ['Open', 'High', 'Low', 'Close']:
                standard_slice_df[column] = (slice_df[column] / slice_df['Close'].shift(1) - 1) * scale
            standard_slice_df = pd.concat([standard_slice_df, standard_slice_df_volume], axis=1)
            standard_slice_df = standard_slice_df.fillna(0)

            standard_slice_df = standard_slice_df[0:self.full_window_size]
            standard_obs_df = standard_slice_df[0:self.window_size]
            standard_train_df = standard_slice_df[self.window_size:self.full_window_size]

        return standard_slice_df, standard_obs_df, standard_train_df

    def render(self, render_mode=None):
        font = 'Verdana'

        # action_history에 임의의 값 추가
        # self.action_history = pd.DataFrame({'action': [0,1,2,3,0,1,2,3,0,1]}, index=self.slice_df.index[:10])

        if render_mode == "human":
            candle = go.Candlestick(open=self.slice_df['Open'], high=self.slice_df['High'], low=self.slice_df['Low'], close=self.slice_df['Close'],
                                    increasing_line_color='rgb(38, 166, 154)', increasing_fillcolor='rgb(38, 166, 154)',
                                    decreasing_line_color='rgb(239, 83, 80)', decreasing_fillcolor='rgb(239, 83, 80)', yaxis='y1')
            fig = go.Figure(data=[candle])

            # action DataFrame의 각 행에 대해 반복
            for index, row in self.action_history.iterrows():
                if index in self.slice_df.index:
                    x_position = self.slice_df.index.get_loc(index)  # x축 위치 결정

                    if row['action'] == 0:
                        # 위로 향한 빨간 삼각형
                        fig.add_shape(type="line", x0=x_position, y0=0, x1=x_position, y1=1, xref='x', yref='paper', line=dict(color="rgba(38, 166, 154, 0.3)", width=8))
                    elif row['action'] == 1:
                        # 아래로 향한 파란 삼각형
                        fig.add_shape(type="line", x0=x_position, y0=0, x1=x_position, y1=1, xref='x', yref='paper', line=dict(color="rgba(239, 83, 80, 0.3)", width=8))
                    elif row['action'] == 2:
                        # 초록색 선
                        fig.add_shape(type="line", x0=x_position, y0=0, x1=x_position, y1=1, xref='x', yref='paper', line=dict(color="rgba(0, 255, 0,0.3)", width=8))

            # font = 'Open Sans'
            # font = 'Droid Sans'
            # font = 'PT Sans Narrow'

            # start_step 선과 텍스트 추가
            fig.add_shape(type="line", x0=self.current_index, y0=0, x1=self.current_index, y1=1, xref='x', yref='paper', line=dict(color="rgb(255, 183, 77)", width=1))

            fig.add_shape(type="line", x0=self.start_step, y0=0, x1=self.start_step, y1=1, xref='x', yref='paper', line=dict(color="rgb(255, 183, 77)", width=1))

            fig.add_annotation(x=self.start_step, y=1, text="Start", showarrow=True, arrowhead=1, xref="x", yref="paper", arrowcolor="rgb(255, 183, 77)", arrowsize=1.1, arrowwidth=2, ax=-20, ay=-30,
                               font=dict(family=font, size=12, color="rgb(255, 183, 77)"), align="center")

            fig.add_annotation(x=self.current_index, y=1, text="Now", showarrow=True, arrowhead=1, xref="x", yref="paper", arrowcolor="rgb(255, 183, 77)", arrowsize=1.1, arrowwidth=2, ax=20, ay=-30,
                               font=dict(family=font, size=12, color="rgb(255, 183, 77)"), align="center")

            balance_trace = go.Scatter(x=list(range(self.start_step, self.start_step + len(balance_history))), y=balance_history, mode='lines', name='Balance', line=dict(color='rgb(255, 183, 77)'), yaxis='y2')
            fig.add_trace(balance_trace)
            # 레이아웃 업데이트
            fig.update_layout(
                height=900,
                width=1800,
                plot_bgcolor='rgb(13, 14, 20)',
                xaxis=dict(domain=[0, 1]),
                yaxis=dict(title='Net Worth', side='left', overlaying='y2'),
                yaxis2=dict(title='Balance', side='right', range=[min(balance_history), max(balance_history)]),
                title='RL 차트',
                template='plotly_dark'
            )

            fig.show()


# if __name__ == "__main__":
#     df_path = "D:\AI_Learning/bitcoin_1m_data_2021-01-01_to_2024-03-19.csv"  # 각자 .csv파일 경로 지정하는 식으로 (ex:"D:\AI_Learning/bitcoin_chart_Data.csv" )
#     df = pd.read_csv(df_path)
#     try:
#         df = df.drop(['Date'], axis=1)
#     except:
#         pass
#
#     action_history = []
#     balance_history = []
#     current_price_history = []
#
#
#     def make_env(env_id, rank, full_window_size, test_window_size, seed=0):
#         def _init():
#             env = stablebaselineEnv(df, full_window_size, test_window_size)
#             env.seed(seed + rank)
#             return env
#
#         set_random_seed(seed)
#         return _init
#

    # def objective(trial, df):
    #     # Optuna를 사용하여 하이퍼 파라미터 설정
    #     full_window_size = trial.suggest_int("full_window_size", 500, 1000)
    #     test_window_size = trial.suggest_int("test_window_size", 100, 400)
    #     num_cpu = trial.suggest_int("num_cpu", 1, 12)
    #     eval_freq = trial.suggest_int("eval_freq", 500, 2000)
    #
    #     # 신경망 모델 하이퍼 파라미터
    #     learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    #     batch_size = trial.suggest_int("batch_size", 32, 256)
    #     n_steps = trial.suggest_int("n_steps", 16, 128)
    #     gamma = trial.suggest_uniform("gamma", 0.9, 0.9999)
    #     gae_lambda = trial.suggest_uniform("gae_lambda", 0.9, 0.9999)
    #     cg_max_steps = trial.suggest_int("cg_max_steps", 5, 20)
    #     cg_damping = trial.suggest_loguniform("cg_damping", 1e-5, 1e-1)
    #     line_search_shrinking_factor = trial.suggest_uniform("line_search_shrinking_factor", 0.5, 0.9)
    #     line_search_max_iter = trial.suggest_int("line_search_max_iter", 5, 20)
    #     # ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 1e-1)
    #     # vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    #     # max_grad_norm = trial.suggest_uniform("max_grad_norm", 0.3, 5)
    #
    #     # 환경 생성
    #     env = SubprocVecEnv([make_env(0, i, full_window_size, test_window_size) for i in range(num_cpu)])
    #     # env = Monitor(env)  # 기존 환경을 Monitor로 감싸기
    #
    #     # 모델 생성
    #     # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard/",
    #     #             learning_rate=learning_rate, batch_size=batch_size, n_steps=n_steps,
    #     #             gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef,
    #     #             max_grad_norm=max_grad_norm)
    #     model = TRPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard/",
    #                  learning_rate=learning_rate, batch_size=batch_size, n_steps=n_steps,
    #                  gamma=gamma, gae_lambda=gae_lambda,
    #                  cg_max_steps=cg_max_steps, cg_damping=cg_damping,
    #                  line_search_shrinking_factor=line_search_shrinking_factor,
    #                  line_search_max_iter=line_search_max_iter)
    #
    #     # 평가 콜백 정의
    #     eval_callback = EvalCallback(env, best_model_save_path='./best_model', log_path='./logs',
    #                                  eval_freq=eval_freq, deterministic=True, render=False,
    #                                  verbose=1)
    #
    #     # 모델 학습
    #     model.learn(total_timesteps=3e6, callback=eval_callback) # 실행-> tensorboard --logdir=./tensorboard/
    #
    #     # 최적화 목적: 누적 수익률
    #     obs = env.reset()
    #     done = [False] * num_cpu  # done을 환경 개수만큼의 리스트로 초기화
    #     total_pnl = 0
    #     while not all(done):
    #         action, _states = model.predict(obs)
    #         obs, reward, done, info = env.step(action)
    #
    #         for i in range(num_cpu):
    #             if not done[i]:
    #                 total_pnl += info[i]['total_pnl']
    #
    #     total_return = total_pnl / num_cpu  # 모든 환경의 평균 total_pnl을 계산
    #
    #     print(f"Trial {trial.number}")
    #     print("Hyperparameters:")
    #     print(trial.params)
    #     print(f"Eval/Mean Reward: {eval_callback.last_mean_reward}")
    #     print("--------------------")
    #     # 현재 시도의 하이퍼 파라미터 값과 eval/mean_reward 값을 딕셔너리로 저장
    #     trial_results = {
    #         'trial_number': trial.number,
    #         'eval_mean_reward': eval_callback.last_mean_reward,
    #         'total_return': total_return,
    #         **trial.params
    #     }    # CSV 파일에 결과 저장
    #     with open('hyperparameter_results.csv', 'a', newline='') as file:
    #         fieldnames = list(trial_results.keys())
    #         writer = csv.DictWriter(file, fieldnames=fieldnames)
    #
    #         if trial.number == 0:
    #             writer.writeheader()  # 첫 번째 시도인 경우 헤더 작성
    #         writer.writerow(trial_results)
    #
    #     return total_return
    #
    #
    # # Optuna를 사용하여 하이퍼 파라미터 튜닝
    # study = optuna.create_study(direction="maximize")
    # study.optimize(lambda trial: objective(trial, df), n_trials=10)
    #
    # print("Best hyperparameters:", study.best_params)
    # print("Best total return:", study.best_value)
    #
    #
    #


if __name__ == "__main__":
    df_path = "D:\AI_Learning/bitcoin_1m_data_2021-01-01_to_2024-03-19.csv"  # 각자 .csv파일 경로 지정하는 식으로 (ex:"D:\AI_Learning/bitcoin_chart_Data.csv" )
    df = pd.read_csv(df_path)
    try:
        df = df.drop(['Date'], axis=1)
    except:
        pass

    action_history = []
    balance_history = []
    current_price_history = []


    def make_env(env_id, rank, full_window_size, test_window_size, seed=0):
        def _init():
            env = stablebaselineEnv(df, full_window_size, test_window_size)
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init


    def objective(trial, df):
        env = SubprocVecEnv([make_env(0, i, 846, 358) for i in range(6)])
        # env = Monitor(env)  # 기존 환경을 Monitor로 감싸기

        # 모델 생성
        # model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tensorboard/",
        #             learning_rate=learning_rate, batch_size=batch_size, n_steps=n_steps,
        #             gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef,
        #             max_grad_norm=max_grad_norm)
        model = TRPO("MultiInputPolicy", env, verbose=0, tensorboard_log="./tensorboard/",
                     learning_rate=1.581632489885157e-05, batch_size=113, n_steps=82,
                     gamma=0.9715185625064103, gae_lambda= 0.9896882174341037,
                     cg_max_steps=13, cg_damping=0.022827241624845087,
                     line_search_shrinking_factor=0.7249390105673472,
                     line_search_max_iter=18)

        # 평가 콜백 정의
        eval_callback = EvalCallback(env, best_model_save_path='./best_model', log_path='./logs',
                                     eval_freq=1472, deterministic=True, render=False,
                                     verbose=1)

        # 모델 학습
        model.learn(total_timesteps=5e6, callback=eval_callback) # 실행-> tensorboard --logdir=./tensorboard/
        model.save('train_model2')

        # 최적화 목적: 누적 수익률
        obs = env.reset()
        done = [False] * 6  # done을 환경 개수만큼의 리스트로 초기화
        total_pnl = 0
        while not all(done):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)

            for i in range(6):
                if not done[i]:
                    total_pnl += info[i]['total_pnl']

        total_return = total_pnl / 6  # 모든 환경의 평균 total_pnl을 계산

        print(f"Trial {trial.number}")
        print("Hyperparameters:")
        print(trial.params)
        print(f"Eval/Mean Reward: {eval_callback.last_mean_reward}")
        print("--------------------")
        # 현재 시도의 하이퍼 파라미터 값과 eval/mean_reward 값을 딕셔너리로 저장
        trial_results = {
            'trial_number': trial.number,
            'eval_mean_reward': eval_callback.last_mean_reward,
            'total_return': total_return,
            **trial.params
        }    # CSV 파일에 결과 저장
        with open('hyperparameter_results.csv', 'a', newline='') as file:
            fieldnames = list(trial_results.keys())
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if trial.number == 0:
                writer.writeheader()  # 첫 번째 시도인 경우 헤더 작성
            writer.writerow(trial_results)
        return total_return


    # # Optuna를 사용하여 하이퍼 파라미터 튜닝
    # study = optuna.create_study(direction="maximize")
    # study.optimize(lambda trial: objective(trial, df), n_trials=1)
    # #
    # #
    # print("Best hyperparameters:", study.best_params)
    # print("Best total return:", study.best_value)
    # #













    # env = stablebaselineEnv()
    # pd.set_option('display.max_columns', None)  # 모든 열 출력
    # pd.set_option('display.width', None)  # 출력 너비 제한 없음
    # pd.set_option('display.max_rows', None)  # 모든 행 출력

    # print(env.final_slice_df)
    # print(env.final_obs_df)
    # print(env.final_train_df)
    # print(len(env.final_train_df))
    # print(len(env.final_train_df))
    # print(len(env.final_train_df))
    # print(env.standard_obs_df)
    # print(env.standard_train_df)
    # print(env.current_step)
    # print(env.current_price)
    # env.step(1)
    # print(" ")
    # print("next step")
    # print(" ")
    # print(env.get_obs())
    # print(env.current_step)
    # print(env.current_price)
    # env.step(1)
    # print(" ")
    # print("next step")
    # print(" ")
    # print(env.current_step)
    # print(env.current_price)
    # env.step(1)
    # print(" ")
    # print("next step")
    # print(" ")
    # print(env.current_step)
    # print(env.current_price)




    # """
    # 아래 코드만 따로 실행 시 에측 결과 확인
    # """

    full_window_size = 846  # slice_df의 크기 / 자른 데이터의 전체 크기
    test_window_size = 358   # 에이전트가 볼 수 없고 학습을 진행해야 하는 차트의 크기
    env = stablebaselineEnv(df,full_window_size,test_window_size)
    model = TRPO.load('train_model2.zip')
    # model.load('train_model.zip')
    #
    # 환경 초기화
    obs = env.reset()
    done = False


    # 에피소드 내에서 스텝 반복
    while not done:
        # 모델을 사용하여 행동 예측
        action, _states = model.predict(obs)

        # 선택한 액션 저장
        action_history.append(action)

        # 환경에서 한 스텝 실행
        obs, reward, done, info = env.step(action)

        # 현재 잔고 저장, 현재 가격 저장
        balance_history.append(env.pnl)
        current_price_history.append(env.current_price)

    # 환경 렌더링
    env.render("human")

    # 환경 종료
    env.close()

    # 액션 히스토리와 잔고 히스토리 출력
    print("Action History:")
    for i, action in enumerate(action_history):
        print(f"Step {i + 1}: Action={action}, Balance={balance_history[i]}, current_price = {current_price_history[i]}")
