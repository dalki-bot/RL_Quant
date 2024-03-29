import pandas as pd
import numpy as np
import random
import mplfinance as mpf
import shimmy
import ta
import plotly.graph_objects as go
from collections import deque
import gym
from gym import spaces
import stable_baselines3
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

class stablebaselineEnv(gym.Env):
    def __init__(self, df, full_window_size, obs_window_size, usdt_balance, btc_size=0, leverage=1): 
        super(stablebaselineEnv, self).__init__()
        
        # 데이터 처리 변수
        self.df = df
        self.full_window_size = full_window_size
        self.obs_window_size = obs_window_size
        self.start_idx = 0
        self.full_window, self.obs_window = self.generate_random_data_slice()
        self.current_step = self.obs_window.tail(1)
        self.current_price = self.get_price()
        
        # action, obs 공간 정의
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "chart_data": spaces.Box(low=0, high=np.inf, shape=self.obs_window.shape, dtype=np.float32), # 차트 데이터
            "position": spaces.Discrete(3),  # 포지션 {0:Long, 1:Short, 2:None}
            "action": spaces.Discrete(4),  # 액션 {0:Long, 1:Short, 2:Close, 3:Hold}
            "current_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 현재 가격
            "avg_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 평균 진입 가격
            "pnl": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),  # 미실현 손익
            "total_pnl": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),  # 누적 손익
            "usdt_balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # USDT 잔고
            "size": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 포지션 수량
            "margin": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),  # 사용 중인 마진
            "total_balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)  # 총 자산
        })
        
        # 트레이딩 관련 변수
        self.initial_usdt_balance = usdt_balance # 초기 usdt 잔고
        self.min_order = 0.002 # 최소 주문 수량
        self.fee = 0.0005 # 거래 수수료
        self.leverage = leverage # 레버리지
        self.usdt_balance = usdt_balance # 초기 usdt 잔고
        self.btc_size = btc_size # 포지션 수량
        self.margin = 0 # 포지션 증거금
        self.position = 2 # 포지션 {0:Long, 1:Short, 2:None}
        self.order_price = 0 # 주문 금액
        self.current_avg_price = 0 # 현재 평단가
        self.unrealized_pnl = 0 # 미실현 손익
        self.realized_pnl = 0 # 실현 손익
        self.max_pnl = 0 # 최대 손익
        self.total_balance = usdt_balance # 총 자산
        self.previous_total_balance = usdt_balance  # 이전 총 자산
        
        # 액션 관련 변수
        self.hold_count = 0
        
        # 랜더링 관련 변수
        self.action_history = pd.DataFrame(columns=['step', 'action'])
        
    def reset(self):
        self.start_idx = 0
        self.full_window, self.obs_window = self.generate_random_data_slice()
        self.current_step = self.obs_window.tail(1)
        self.current_price = self.get_price()
    
        self.leverage = self.leverage # 레버리지
        self.usdt_balance = self.initial_usdt_balance # 초기 usdt 잔고
        self.btc_size = self.btc_size # 포지션 수량
        self.margin = 0 # 포지션 증거금
        self.position = 2 # 포지션 {0:Long, 1:Short, 2:None}
        self.order_price = 0 # 주문 금액
        self.current_avg_price = 0 # 현재 평단가
        self.unrealized_pnl = 0 # 미실현 손익
        self.realized_pnl = 0 # 실현 손익
        self.max_pnl = 0 # 최대 손익
        self.total_balance = self.initial_usdt_balance # 총 자산
        self.previous_total_balance = self.initial_usdt_balance  # 이전 총 자산
        
        self.action_history = pd.DataFrame(columns=['step', 'action'])
        return self.get_obs()
        
    def generate_random_data_slice(self):
        start_winodow = np.random.randint(0, len(self.df) - self.full_window_size)
        full_window = self.df[start_winodow:start_winodow + self.full_window_size].reset_index(drop=True)
        obs_window = full_window[self.start_idx:self.start_idx + self.obs_window_size]
        return full_window, obs_window
    
    def next_obs(self):
        self.start_idx += 1
        self.obs_window = self.full_window[self.start_idx:self.start_idx + self.obs_window_size]
        self.current_step = self.obs_window.tail(1)
            
    def get_price(self):
        open = self.full_window.iloc[self.obs_window.index[-1]+1]['Open']
        close = self.full_window.iloc[self.obs_window.index[-1]+1]['Close']
        self.current_price = round(random.uniform(open, close), 2)
        return self.current_price

    def reward(self, action):
        reward = 0
        if self.unrealized_pnl > 0:
            reward += self.unrealized_pnl
        else:
            reward += self.unrealized_pnl * 2
            
        if action == 2 and self.realized_pnl > 0:
            reward += self.realized_pnl

        if action == 2:
            reward = self.realized_pnl - self.max_pnl
        else:
            pass
        
        return reward

    def act(self, action):
        # 행동
        self.current_price = self.get_price()
        size = round(self.usdt_balance / self.current_price, 3)
        if (action == 0 or action == 1) and self.position == 2:
            if self.usdt_balance > size * self.current_price:
                fee = size * self.fee
                self.position_size += size
                self.margin += (size * self.current_price) / self.leverage
                self.usdt_balance -= self.margin + fee
                self.position = action
            else:
                action = 3
            
        elif action == 2 and self.position != 2:
            fee = self.btc_size * self.fee
            self.usdt_balance += (self.position_size * self.current_price) - fee
            self.position_size = 0
            self.margin = 0
            self.position = action
        else:
            action = 3
        
        # 포트폴리오 가치 계산
        self.current_avg_price = ((self.current_avg_price * (self.btc_size - size)) + (self.current_price * size)) / self.btc_size # 평단가 계산
        
        if self.current_avg_price != 0: # 미실현 손익 계산
            self.unrealized_pnl = round((self.current_price - self.current_avg_price) / self.current_avg_price * 100, 2)
            self.max_pnl = max(self.max_pnl, self.unrealized_pnl)
        else:
            self.unrealized_pnl = 0
            self.max_pnl = 0
            
        if action == 2: # 실현 손익 계산
            self.realized_pnl = self.unrealized_pnl
        else:
            self.realized_pnl = 0
            
        self.total_balance = self.usdt_balance + self.margin # 총 자산 계산
        
        reward = self.reward(action) # 보상 계산
        return action, reward
    
    " act : 행동 -> 평단가 계산 -> 미실현 손익 계산 -> 실현 손익 계산 -> 총 자산 계산 -> 보상 계산 "
        

            
        
            

    def get_obs(self, action=3):
        obs = {
            "chart_data": self.obs_window.values,
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

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def step(self, action):
        self.get_price()
        action = self.act(action)
        self.next_obs()
        self.action_history.loc[len(self.action_history)] = [self.obs_window.index[-1], action]
        if self.obs_window.index[-1] == self.full_window.index[-1]:
            done = True
        else:
            done = False
        info = {}
        reward = 0
        if self.total_balance <= self.previous_total_balance:
            reward -= 1
        else:
            reward += 1
        self.previous_total_balance = self.total_balance
        if self.hold_count > 50:
            reward -= 2
        return self.get_obs(), reward, done, info
    
    def render(self, render_mode=None):
        font = 'Verdana'
        if render_mode == "human":
            candle = go.Candlestick(open=self.full_window['Open'], high=self.full_window['High'], low=self.full_window['Low'], close=self.full_window['Close'],
                                    increasing_line_color='rgb(38, 166, 154)', increasing_fillcolor='rgb(38, 166, 154)',
                                    decreasing_line_color='rgb(239, 83, 80)', decreasing_fillcolor='rgb(239, 83, 80)', yaxis='y2')
            fig = go.Figure(data=[candle])

            # action DataFrame의 각 행에 대해 반복
            for _, row in self.action_history.iterrows():
                step = row['step']
                if step in self.full_window.index:
                    x_position = self.full_window.index.get_loc(step)  # x축 위치 결정

                    if row['action'] == 0:
                        # 위로 향한 빨간 삼각형
                        fig.add_shape(type="line", x0=x_position, y0=0, x1=x_position, y1=1, xref='x', yref='paper', line=dict(color="rgba(38, 166, 154, 0.3)", width=8))
                    elif row['action'] == 1:
                        # 아래로 향한 파란 삼각형
                        fig.add_shape(type="line", x0=x_position, y0=0, x1=x_position, y1=1, xref='x', yref='paper', line=dict(color="rgba(239, 83, 80, 0.3)", width=8))
                    elif row['action'] == 2:
                        # 초록색 선
                        fig.add_shape(type="line", x0=x_position, y0=0, x1=x_position, y1=1, xref='x', yref='paper', line=dict(color="rgba(0, 255, 0,0.3)", width=8))

            # 시작선
            fig.add_shape(type="line", x0=self.obs_window_size, y0=0, x1=self.obs_window_size, y1=1, xref='x', yref='paper', line=dict(color="rgb(255, 183, 77)", width=1))
            fig.add_annotation(x=self.obs_window_size, y=1, text="Start", showarrow=True, arrowhead=1, xref="x", yref="paper", arrowcolor="rgb(255, 183, 77)", arrowsize=1.1, arrowwidth=2, ax=-20, ay=-30,
                                font=dict(family=font, size=12, color="rgb(255, 183, 77)"), align="center")

            # 현재 step 선
            fig.add_shape(type="line", x0=self.obs_window.index[-1], y0=0, x1=self.obs_window.index[-1], y1=1, xref='x', yref='paper', line=dict(color="rgb(255, 183, 77)", width=1))
            fig.add_annotation(x=self.obs_window.index[-1], y=1, text="Now", showarrow=True, arrowhead=1, xref="x", yref="paper", arrowcolor="rgb(255, 183, 77)", arrowsize=1.1, arrowwidth=2, ax=20, ay=-30,
                                font=dict(family=font, size=12, color="rgb(255, 183, 77)"), align="center")

            # 레이아웃 업데이트
            fig.update_layout(
                height=900,
                width=1800,
                plot_bgcolor='rgb(13, 14, 20)',
                xaxis=dict(domain=[0, 1]),
                yaxis=dict(title='Net Worth', side='right', overlaying='y2'),
                yaxis2=dict(title='Price', side='left'),
                title='RL 차트',
                template='plotly_dark'
            )

            fig.show()

class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq):
        super().__init__()
        self.env = env
        self.render_freq = render_freq
        self.episode_count = 0

    def _on_step(self):
        if self.n_calls % self.render_freq == 0:
            self.env.render(render_mode="human")
        return True

# df = pd.read_csv(r'C:\Users\user\Documents\GitHub\RL_Quant\btctest.csv') # 회사
df = pd.read_csv(r'C:\Users\dyd46\Documents\GitHub\RL_Quant\btctest.csv') # 집
full_window_size = 200
obs_window_size = 50
env = stablebaselineEnv(df, full_window_size, obs_window_size, usdt_balance=1000)

model = PPO("MultiInputPolicy", env, verbose=1)
obs = env.reset()
render_callback = RenderCallback(env, render_freq=149*10)

model.learn(total_timesteps=150*1000)
# , callback=render_callback