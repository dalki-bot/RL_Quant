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
    def __init__(self, df, full_window_size, test_window_size, usdt_balance=1000, btc_size=0, leverage=1): 
        super(stablebaselineEnv, self).__init__()
        self.slice_df, self.obs_df, self.train_df = stablebaselineEnv.generate_random_data_slice(df, full_window_size, test_window_size) # 랜덤 위치로 slice된 차트 데이터 초기화
        self.action_space = spaces.Discrete(4)  # 0: Long, 1: Short, 2: Close, 3: Hold
        self.current_step = self.slice_df.tail(1)
        self.observation_space = spaces.Dict({
            "chart_data": spaces.Box(low=0, high=np.inf, shape=(len(self.current_step.columns,)), dtype=np.float32), # 차트 데이터
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

        self.start_step = full_window_size
        self.window_size = full_window_size
        self.test_window_size = test_window_size
        self.current_price = None
        self.df = df

        # reset 미포함
        self.initial_usdt_balance = usdt_balance # 초기 usdt 잔고
        self.min_order = 0.002 # 최소 주문 수량
        self.fee = 0.0005 # 거래 수수료
        self.leverage = leverage # 레버리지
        
        # reset 포함
        self.usdt_balance = usdt_balance # 초기 usdt 잔고
        self.btc_size = btc_size # 포지션 수량
        self.margin = 0 # 포지션 증거금
        self.position = None # 포지션 {0:Long, 1:Short, 2:None}
        self.order_price = 0 # 주문 금액
        self.last_size_value = 0 # (평단가 계산에 필요)
        self.current_avg_price = 0 # 현재 평단가
        self.pnl = 0 # 미실현 손익
        self.closing_pnl = 0 # 실현 손익
        self.total_pnl = 0 # 누적 손익
        self.total_fee = 0 # 누적 수수료
        self.total_balance = 0 # 총 자산
        self.action_history = pd.DataFrame(columns=['action'])
        self.current_index = 0
        pass

    def reset(self): # 리셋 함수 
        self.slice_df, self.obs_df, self.train_df = stablebaselineEnv.generate_random_data_slice(self.df, self.window_size,self.test_window_size) # reset 하며 새로운 랜덤 차트 데이터 초기화
        
        self.usdt_balance = self.initial_usdt_balance # 초기 usdt 잔고
        self.btc_size = 0 # 포지션 수량
        self.margin = 0 # 포지션 증거금
        self.position = None # 포지션 {0:Long, 1:Short, 2:None}
        self.order_price = 0 # 주문 금액
        self.last_size_value = 0 # (평단가 계산에 필요)
        self.current_avg_price = 0 # 현재 평단가
        self.pnl = 0 # 미실현 손익
        self.closing_pnl = 0 # 실현 손익
        self.total_pnl = 0 # 누적 손익
        self.total_fee = 0 # 누적 수수료
        self.total_balance = 0 # 총 자산
        self.action_history = pd.DataFrame(columns=['action'])
        pass
    
    
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
    def generate_random_data_slice(df, full_window_size, test_window_size):
        strat_index = np.random.randint(0, len(df) - full_window_size)
        end_index = strat_index + full_window_size
        obs_end = end_index - test_window_size
    
        # slice_df : 전체 데이터에서 랜덤한 위치로 잘라낸 데이터
        # obs_df : 전체 데이터중 현재 스텝의 이전 데이터로써 이미 알고있는 차트 데이터
        # train_df : 전체 데이터중 현재 스텝의 이후 데이터로써 학습을 위한 차트 데이터
        return slice_df[strat_index:end_index], obs_df[strat_index:obs_end], train_df[obs_end:end_index]

    # 다음 step과 가격을 가져옴
    def next_obs(self): 
        row_to_move = self.train_df.iloc[0:1] # train_df의 첫 행을 가져옴
        self.obs_df = pd.concat([self.obs_df, row_to_move], ignore_index=True) # obs_df의 마지막 행에 train_df의 첫 행을 추가
        self.train_df = self.train_df.drop(self.train_df.index[0]) # train_df의 첫 행을 제거 (메모리 절약을 위해)
        self.current_step = self.obs_df.tail(1) # obs_df의 마지막 행을 현재 스텝으로 설정
        self.current_price = round(random.uniform(self.current_step['Open'].iloc[-1], self.current_step['Close'].iloc[-1]), 2) # 현재 가격을 시가, 종가 사이 랜덤 값으로 결정

    def step(self, action):
        self.next_obs() # 다음 obs를 가져옴
        if (np.array(self.current_step) == None).all(): # 다음 훈련 데이터 없을 시 done = True로 변경 종료
            done = True
            return None, None, done, {}
        
        action = self.act(action) # action을 수행함.
        action_row = pd.DataFrame({'action': [action]}, index=[self.slice_df.index[self.current_index]]) 
        self.action_history = pd.concat([self.action_history, action_row]) # action_history에 action 추가
        
        reward = None
        done = False
        obs = {
            "chart_data": self.current_step,
            "position": self.position,
            "action": action,
            "current_price": self.current_price,
            "avg_price": self.current_avg_price,
            "pnl": self.pnl,
            "total_pnl": self.total_pnl,
            "usdt_balance": round(self.usdt_balance, 2),
            "size": self.btc_size,
            "margin": self.margin,
            "total_balance": self.total_balance
        }
        info = {}
        return obs, reward, done, info

    '''
    스텝 :
        1. 다음 step과 price를 가져옴
        2. 다음 훈련 데이터 없을 시 done = True로 변경 종료
        3. action을 수행함.
        4. action_history에 action 추가
        5. reward : 미구현 ###############################
        6. obs : 현재 step의 관측치 반환 (수정필요)
        
    done 조건 생각해보기

    '''
















    def render(self, render_mode=None):
        font = 'Verdana'

        if render_mode == "human":
            candle = go.Candlestick(open=self.slice_df['Open'], high=self.slice_df['High'], low=self.slice_df['Low'], close=self.slice_df['Close'],
                                    increasing_line_color='rgb(38, 166, 154)', increasing_fillcolor='rgb(38, 166, 154)',
                                    decreasing_line_color='rgb(239, 83, 80)', decreasing_fillcolor='rgb(239, 83, 80)', yaxis='y2')
            fig = go.Figure(data=[candle])

            # action DataFrame의 각 행에 대해 반복
            for index, row in self.action_history.iterrows():
                if index in self.slice_df.index:
                    x_position = self.slice_df.index.get_loc(index)  # x축 위치 결정

                    if row['action'] == 0:
                        # 위로 향한 빨간 삼각형
                        fig.add_shape(type="line", x0=x_position, y0=0, x1=x_position, y1=1, xref='x', yref='paper', line=dict(color="rgba(38, 166, 154, 0.3)", width=5))
                    elif row['action'] == 1:
                        # 아래로 향한 파란 삼각형
                        fig.add_shape(type="line", x0=x_position, y0=0, x1=x_position, y1=1, xref='x', yref='paper', line=dict(color="rgba(239, 83, 80, 0.3)", width=5))
                    elif row['action'] == 2:
                        # 초록색 원
                        fig.add_shape(type="line", x0=x_position, y0=0, x1=x_position, y1=1, xref='x', yref='paper', line=dict(color="rgba(0, 255, 0,0.3)", width=5))

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

            # 레이아웃 업데이트
            fig.update_layout(
                height=600,
                width=1000,
                plot_bgcolor='rgb(13, 14, 20)',
                xaxis=dict(domain=[0, 1]),
                yaxis=dict(title='Net Worth', side='right', overlaying='y2'),
                yaxis2=dict(title='Price', side='left'),
                title='RL 차트',
                template='plotly_dark'
            )

            fig.show()


    # 여러가지 보조지표들을 추가하는 함수, 보조지표들을 넣고 싶다면 여기서 수정 진행
    def add_indicator(self,df):
        # RSI & ADX 보조지표 추가
        df['RSI'] = ta.momentum.rsi(df['Close'])
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])

        # # 단순 이동평균선 추가
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)

        # 지수 이동평균선 추가
        # df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
        # df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)

        # 볼린저 밴드 추가
        bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['Bollinger_High'] = bollinger.bollinger_hband()
        # df['Bollinger_Mid'] = bollinger.bollinger_mavg() #window가 20일 때 20일선과 같은것으로 알고 있음
        df['Bollinger_Low'] = bollinger.bollinger_lband()


    # 노멀라이즈 진행함, 노멀라이즈 진행 시 날짜 정보는 제외됨. 학습에 넣을지 말지는 추후 협의
    
    def normalization(self, df):
        # Date 열 제거
        try:
            df = df.drop(['Date'], axis=1)
        except:
            pass
        none_normalized_df = df

        # Volume 열에 대해 로그 변환 수행 (거래량의 경우 많은날은 너무 값이 튀어서 문제가 생길 수 있다고 판단되어 로그 스케일로 진행)
        try:
            df['Volume'] = np.log(df['Volume'])
        except:
            pass
        # 로그 변환된 Volume과 다른 열들에 대해 정규화 수행
        normalized_df = (df - df.min()) / (df.max() - df.min())

        return normalized_df, none_normalized_df

full_window_size = 100  # slice_df의 크기 / 자른 데이터의 전체 크기
test_window_size = 300  # 에이전트가 볼 수 없고 학습을 진행해야 하는 차트의 크기 

df_path = pd.DataFrame # 각자 .csv파일 경로 지정하는 식으로 (ex:"D:\AI_Learning/bitcoin_chart_Data.csv" )


# # 신경망 모델을 만듬 - MultiInputPolicy(stable baseline에서 신경망 구조 알아서 만들어줌), env: 환경을 받아옴, verbose: log print 양 결정(0:전체,1:심플,2:제외)
# env = make_vec_env(lambda: stablebaselineEnv(), n_envs=1)

# model = PPO("MultiInputPolicy", env, verbose=1) 
# obs = env.reset()



'''
백업용 코드
 # df 데이터를 받아 full_window_size + test_window_size만큼 랜덤 위치로 잘라서 자른 df를 반환해주는 함수
    def generate_random_data_slice(data, full_window_size, test_window_size):
        max_start_index = len(data) - full_window_size - test_window_size
        if max_start_index <= 0:
            raise ValueError("데이터 크기가 너무 작아 분할할 수 없습니다.")

        start_index = np.random.randint(0, max_start_index)
        end_index = start_index + full_window_size
        test_end_index = end_index + test_window_size

        return data[start_index:test_end_index], data[start_index:end_index], data[end_index:test_end_index]

    def next_observation(self):
        obs_df = self.slice_df.iloc[(self.current_index-101)+1:(self.current_index-1)+1]
        return obs_df
'''
