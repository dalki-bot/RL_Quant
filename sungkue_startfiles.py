import random
import pandas as pd
import gym
import stable_baselines3
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
import mplfinance as mpf
import shimmy
import ta
import plotly.graph_objects as go
from collections import deque

"""
할 일

self.observation이 step이 지나감에 따라 같이 update시키기. -- 완료.
    - self.observation의 크기를 유지 하기 위해 window_siz는 유지하며 step을 따라가도록 변경 --완료
    - current_step이 df보다 커지거나 같아질 경우 done = True로 바꾸도록 변경  -- 완료

render 함수 만들기.(가시화)

"""


class stablebaselineEnv(gym.Env):
    def __init__(self, usdt_balance=1000, btc_size=0, leverage=1):
        super(stablebaselineEnv, self).__init__()
        self.df = df
        self.df['Volume'] = np.log(self.df['Volume'])
        self.temp = temp
        self.slice_df, self.obs_df, self.train_df = self.generate_random_data_slice(df, full_window_size, test_window_size)  # 랜덤 위치로 slice된 차트 데이터 초기화

        # self.mean = np.mean(self.obs_df, axis=0)
        # self.std = np.std(self.obs_df, axis=0)
        # self.standard_slice_df, self.standard_obs_df, self.standard_train_df = stablebaselineEnv.standardzation(self.slice_df, self.obs_df, self.train_df, self.mean, self.std,mode=1)  # 정규화를 진행함. 모드 0,1 있음.

        self.action_space = spaces.Discrete(4)  # 0: Long, 1: Short, 2: Close, 3: Hold
        self.observation_space = spaces.Dict({
            "chart_data": spaces.Box(low=0, high=np.inf, shape=self.obs_df.shape, dtype=np.float32),  # 차트 데이터
            # "position": spaces.Discrete(3),  # 포지션 {0:Long, 1:Short, 2:None}
            # "action": spaces.Discrete(4),  # 액션 {0:Long, 1:Short, 2:Close, 3:Hold}
            # "current_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 현재 가격
            # "avg_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 평균 진입 가격
            # "pnl": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),  # 미실현 손익
            # "total_pnl": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),  # 누적 손익
            # "usdt_balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # USDT 잔고
            # "size": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 포지션 수량
            # "margin": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 사용 중인 마진
            # "total_balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)  # 총 자산
        })
        self.full_window_size = full_window_size
        self.test_window_size = test_window_size

        # self.start_step = len(self.full_window_size - self.test_window_size)
        self.current_step = self.obs_df.tail(1)
        self.current_price = round(random.uniform(self.train_df['Open'].iloc[0], self.train_df['Close'].iloc[0]), 2)  # 현재 가격을 시가, 종가 사이 랜덤 값으로 결정

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
        # self.action_history = pd.DataFrame(columns=['action'])
        self.current_index = 0

    def reset(self):  # 리셋 함수
        self.slice_df, self.obs_df, self.train_df = self.generate_random_data_slice(self.df, self.full_window_size, self.test_window_size)  # 랜덤 위치로 slice된 차트 데이터 초기화
        # self.mean = np.mean(self.obs_df, axis=0)
        # self.std = np.std(self.obs_df, axis=0)
        # self.standard_slice_df, self.standard_obs_df, self.standard_train_df = stablebaselineEnv.standardzation(self.slice_df, self.obs_df, self.train_df, self.mean, self.std,mode=1)  # 정규화를 진행함. 모드 0,1 있음.

        # self.start_step = len(self.full_window_size - self.test_window_size)
        self.current_step = self.obs_df.tail(1)
        self.current_price = round(random.uniform(self.train_df.iloc[0]['Open'], self.train_df.iloc[0]['Close']), 2)  # 현재 가격을 시가, 종가 사이 랜덤 값으로 결정

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
        # self.action_history = pd.DataFrame(columns=['action'])
        self.current_index = 0

        return self.get_obs()

    def seed(self, seed=None):
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
        # slice_df : 전체 데이터에서 랜덤한 위치로 잘라낸 데이터
        # obs_df : 전체 데이터중 현재 스텝의 이전 데이터로써 이미 알고있는 차트 데이터
        # train_df : 전체 데이터중 현재 스텝의 이후 데이터로써 학습을 위한 차트 데이터

        return slice_df, obs_df, train_df

    # 다음 step과 가격을 가져옴
    def next_obs(self):
        row_to_move = self.train_df.iloc[0:1] # train_df의 첫 행을 가져옴
        self.obs_df = pd.concat([self.obs_df, row_to_move]) # obs_df의 마지막 행에 train_df의 첫 행을 추가
        self.obs_df = self.obs_df.drop(self.obs_df.index[0]) # obs_df의 첫 행을 제거 (메모리 절약을 위해)
        self.train_df = self.train_df.drop(self.train_df.index[0]) # train_df의 첫 행을 제거 (메모리 절약을 위해)
        self.current_step = self.obs_df.tail(1)  # obs_df의 마지막 행을 현재 스텝으로 설정


    def get_price(self):
        open = self.train_df.iloc[0]['Open']
        close = self.train_df.iloc[0]['Close']
        self.current_price = round(random.uniform(open, close), 2)  # 현재 가격을 시가, 종가 사이 랜덤 값으로 결정

    def calculate_reward(self):
        reward = 0
        if self.closing_pnl > 0:
            reward += 2

        elif self.closing_pnl < 0:
            reward -= 1

        self.closing_pnl = 0
        return reward

    def get_obs(self, action=3):
        obs = {
            "chart_data": self.obs_df.values,
            "position": self.position,
            "action": action,
            "current_price": np.array([self.current_price]),
            "avg_price": np.array([self.current_avg_price]),
            "pnl": np.array([self.pnl]),
            "total_pnl": np.array([self.total_pnl]),
            "usdt_balance": np.array([self.usdt_balance]),
            "size": np.array([self.btc_size]),
            "margin": np.array([self.margin]),
            "total_balance": np.array([self.total_balance])
        }
        return obs

    def step(self, action):
        self.get_price()  # 현재 가격을 가져옴
        self.next_obs()  # 다음 obs를 가져옴
        action = self.act(action)  # action을 수행함.
        reward = self.calculate_reward()  # reward 계산
        # action_row = pd.DataFrame({'action': [action]}, index=[self.slice_df.index[self.current_index]])
        # self.action_history = pd.concat([self.action_history, action_row]) # action_history에 action 추가

        if window_size >= len(self.obs_df)-1:
            done = True
        else:
            done = False

        # if self.total_balance < self.total_balance * 0.3:
        #     done = True
        # else:
        #     done = False

        return self.get_obs(action), reward, done, {}


if __name__ == "__main__":
    full_window_size = 400  # slice_df의 크기 / 자른 데이터의 전체 크기
    test_window_size = 300  # 에이전트가 볼 수 없고 학습을 진행해야 하는 차트의 크기
    window_size = full_window_size - test_window_size
    temp = 0
    df_path = "temp_test.csv"  # 각자 .csv파일 경로 지정하는 식으로 (ex:"D:\AI_Learning/bitcoin_chart_Data.csv" )
    df = pd.read_csv(df_path)
    try:
        df = df.drop(['Date'], axis=1)
    except:
        pass
    # 신경망 모델을 만듬 - MultiInputPolicy(stable baseline에서 신경망 구조 알아서 만들어줌), env: 환경을 받아옴, verbose: log print 양 결정(0:전체,1:심플,2:제외)
    # env = make_vec_env(lambda: stablebaselineEnv(df, full_window_size, test_window_size) , n_envs=1)
    env = stablebaselineEnv()

    print(env.slice_df)
    print(env.obs_df)
    print(env.obs_df)
    print(env.train_df)
    print(env.current_step)

    model = PPO("MultiInputPolicy", env, verbose=1)
    obs = env.reset()
    model.learn(total_timesteps=1000000)