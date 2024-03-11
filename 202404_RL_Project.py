import random
import pandas as pd
import gym
import stable_baselines3
from gym import spaces
import numpy as np
from stable_baselines3 import PPO,DQN,A2C
from stable_baselines3.common.env_util import make_vec_env
import mplfinance as mpf
import ta

class stablebaselineEnv(gym.Env):
    def __init__(self,df, window_size, test_window_size, usdt_balance=1000, btc_size=0, leverage=1): 
        super(stablebaselineEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # 0: Long, 1: Short, 2: Close, 3: Hold
        self.observation_space = spaces.Dict({
            "action": spaces.Discrete(4), # 행동 {0:Long, 1:Short, 2:Close, 3:Hold}
            "position": spaces.Discrete(3), # 포지션 {0:Long, 1:Short, 2:None}
            "chart_data": spaces.Box(low=0, high=np.inf, shape=(df.shape[0], df.shape[1]), dtype=np.float32),
            "current_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # 현재 가격
            "pnl": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # 미실현 손익
            "closing_pnl": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # 실현 손익
            "total_pnl": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # 누적 손익
            "total_balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32) # 총 자산

        })
    
        self.slice_df, self.observation_df, self.train_df = stablebaselineEnv.generate_random_data_slice(df, window_size,test_window_size) # 랜덤 위치로 slice된 차트 데이터 초기화
        self.current_step = self.slice_df.iloc[window_size-2] #음.. 위치가 애매함-> 이 상태로 설정시 current_step은 self.observation_df의 마지막 행에 해당됨.
        """
        할 일

        self.observation이 step이 지나감에 따라 같이 update시키기. -- 완료.
            - self.observation의 크기를 유지 하기 위해 window_siz는 유지하며 step을 따라가도록 변경 --완료
            - current_step이 df보다 커지거나 같아질 경우 done = True로 바꾸도록 변경  -- 완료

        render 함수 만들기.(가시화)
        
                
        """

        #self.portfolio = None ## 추후 수정 예정
        self.usdt_balance = usdt_balance # 초기 usdt 잔고
        self.initial_usdt_balance = usdt_balance 
        self.btc_size = btc_size # 포지션 수량
        self.min_order = 0.002 # 최소 주문 수량
        self.fee = 0.0005 # 거래 수수료
        self.leverage = leverage # 레버리지
        self.margin = 0 # 포지션 증거금
        self.position = 0 # 포지션 {0:Long, 1:Short, 2:None}

        self.order_price = 0 # 주문 금액
        self.last_size_value = 0 # (평단가 계산에 필요)
        self.current_avg_price = 0 # 현재 평단가

        self.pnl = 0 # 미실현 손익
        self.closing_pnl = 0 # 실현 손익
        self.total_pnl = 0 # 누적 손익
        self.total_balance = 0 # 총 자산

    def reset(self): # 리셋 함수 -> ㅇ
        self.slice_df = stablebaselineEnv.generate_random_data_slice(self.df, self.window_size,self.test_window_size) # reset 하며 새로운 랜덤 차트 데이터 초기화
        self.observation_space = spaces.Dict({
            "action": spaces.Discrete(4), # 행동 {0:Long, 1:Short, 2:Close, 3:Hold}
            "position": spaces.Discrete(3), # 포지션 {0:Long, 1:Short, 2:None}
            "chart_data": spaces.Box(low=0, high=np.inf, shape=(df.shape[0], df.shape[1]), dtype=np.float32),
            "current_price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # 현재 가격
            "pnl": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # 미실현 손익
            "closing_pnl": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # 실현 손익
            "total_pnl": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # 누적 손익
            "total_balance": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32) # 총 자산       
             })
        pass
    

    # action = 0: Long, 1: Short, 2: Close, 3: Hold
    # position =  0: Long_position, 1: Short_position, 2: None_position
    # ex) (0.002*68000)/1=136, (0.002*68000)/2=68 필요 증거금 계산 예시 #
    def act_check(self, action): # action을 수행할 수 있는 최소한의 조건 확인
        current_price = self.current_price
        min_order = (self.min_order * current_price) / self.leverage # 레버리지 포함 주문수량
        fee = min_order * self.fee
        usdt_margin_balance = self.usdt_balance + self.margin
        if action == 0: # Long
            if self.position == 0 or 2:
                if self.usdt_balance > min_order + fee: # 현재 usdt만 계산
                    return 0
                else:
                    return 3
            elif self.position == 1:
                if usdt_margin_balance > min_order + fee: # 반대 포지션 청산후 추가되는 증거금 포함해서 계산
                    return 0
                else:
                    return 3

        elif action == 1: # Short
            if self.position == 1 or 2:
                if self.usdt_balance > min_order + fee:
                    return 1
                else:
                    return 3
                
            elif self.position == 0:
                if usdt_margin_balance > min_order + fee:
                    return 1
                else:
                    return 3
            
        elif action == 2: # Close
            if self.position == 2:
                return 3
            
        else: # Hold
            return 3

    def act(self, action):
        current_price = self.current_price # 현재 가격
        self.order_price = (self.min_order * current_price) # 주문 금액
        "추후 진입 수량 변경시에 self.order_price만 수정하면 될것같네요"
        order_margin_price = self.order_price * self.leverage # 레버리지 포함 금액 증거금 ex) (0.002*68000)*1=136, (0.002*68000)*2=68
        action = self.act_check(action)
        self.action = action
        if action == 0: # Long
            if self.position == 0 or 2: # Long or None 포지션시에 진입
                self.btc_size += self.min_order # 포지션수 증가
                self.margin += self.order_price # 증거금 증가
                self.usdt_balance -= order_margin_price + (order_margin_price * self.fee) # 잔고 차감 (수수료 및 증거금)

                self.current_avg_price = (self.last_size_value + self.order_price) / self.btc_size
                self.last_size_value = self.btc_size * current_price

                self.pnl = (self.btc_size * self.current_avg_price) - (self.btc_size * current_price)
                self.closing_pnl = 0
                self.total_pnl += self.closing_pnl
                self.total_balance = self.usdt_balance + self.margin
                self.position = 0
                pass
            
            elif self.position == 1: # Short 포지션시에 진입
                close_fee = (self.btc_size * current_price * self.fee) # 포지션 청산
                self.usdt_balance += self.margin + self.pnl - close_fee # 증거금, 실현 손익, 수수료
                self.margin = 0
                self.btc_size = 0
                self.closing_pnl = self.pnl - close_fee
                "보유 포지션 청산"
                
                self.btc_size += self.min_order # 포지션수 증가
                self.margin += self.order_price # 증거금 증가
                self.usdt_balance -= order_margin_price + (order_margin_price * self.fee) # 잔고 차감 (수수료 및 증거금)

                self.current_avg_price = (self.last_size_value + self.order_price) / self.btc_size
                self.last_size_value = self.btc_size * current_price

                self.pnl = (self.btc_size * self.current_avg_price) - (self.btc_size * current_price)
                self.total_pnl += self.closing_pnl
                self.total_balance = self.usdt_balance + self.margin
                self.position = 0
                "새로운 포지션 진입"
                pass
            
        elif action == 1 # Short
            if self.position == 1 or 2:
                self.btc_size += self.min_order # 포지션수 증가
                self.margin += self.order_price # 증거금 증가
                self.usdt_balance -= order_margin_price + (order_margin_price * self.fee) # 잔고 차감 (수수료 및 증거금)

                self.current_avg_price = (self.last_size_value + self.order_price) / self.btc_size
                self.last_size_value = self.btc_size * current_price

                self.pnl = (self.btc_size * self.current_avg_price) - (self.btc_size * current_price)
                self.closing_pnl = 0
                self.total_pnl += self.closing_pnl
                self.total_balance = self.usdt_balance + self.margin
                self.position = 1
                pass
            elif self.position == 0:
                close_fee = (self.btc_size * current_price * self.fee) # 포지션 청산
                self.usdt_balance += self.margin + self.pnl - close_fee # 증거금, 실현 손익, 수수료
                self.margin = 0
                self.btc_size = 0
                self.closing_pnl = self.pnl - close_fee
                "보유 포지션 청산"
                
                self.btc_size += self.min_order # 포지션수 증가
                self.margin += self.order_price # 증거금 증가
                self.usdt_balance -= order_margin_price + (order_margin_price * self.fee) # 잔고 차감 (수수료 및 증거금)

                self.current_avg_price = (self.last_size_value + self.order_price) / self.btc_size
                self.last_size_value = self.btc_size * current_price

                self.pnl = (self.btc_size * self.current_avg_price) - (self.btc_size * current_price)
                self.total_pnl += self.closing_pnl
                self.total_balance = self.usdt_balance + self.margin
                self.position = 1
                "새로운 포지션 진입"
                pass

        elif action == 2: # Close
            close_fee = (self.btc_size * current_price * self.fee)
            self.usdt_balance += self.margin
            self.usdt_balance -= close_fee
            self.margin = 0
            self.btc_size = 0
            
            self.closing_pnl = self.pnl - close_fee
            self.pnl = 0
            self.total_pnl += self.closing_pnl
            self.total_balance = self.usdt_balance + self.margin
            self.position = 2
            pass
        if action == 3: # Hold
            if self.position == 0 or 1: # 포지션 보유중
                self.pnl = (self.btc_size * self.current_avg_price) - (self.btc_size * current_price)
                self.closing_pnl = 0
                self.total_pnl += self.closing_pnl
                self.total_balance = self.usdt_balance + self.margin
                pass
            
            elif self.position == 2: # 포지션 없음
                self.pnl = 0
                self.closing_pnl = 0
                self.total_pnl += self.closing_pnl
                self.total_balance = self.usdt_balance + self.margin
                pass
            
        return self.position, self.action, self.pnl, self.closing_pnl, self.total_pnl, self.total_balance
        
        '''
        주문 수량은 일단 항상 최소 주문 금액으로 하겠습니다.
        최소 수량으로 해도 0.002개 이고 1배율일 경우 증거금 136usdt 정도 들어갑니다.
        
        추후 수량이 커질시 미결손실 또한 고려해야함
        '''


    def step(self, action):
      
        self.current_price = random.uniform(
            self.slice_df.loc[self.current_step, 'Open'],
            self.slice_df.loc[self.current_step, 'Close']
            ) #현재 가격을 시가, 종가 사이 랜덤 값으로 결정됨.
        



        reward = None

        self.current_index += 1  # 현재 위치를 다음 스텝으로 옮김
        self.current_step = self.slice_df.iloc[self.current_index]
        self.observation_df = self.next_observation()

        if self.current_index >=  (self.test_window_size + self.window_size)-1: # 현재 위치가 window_size + test_window_size만큼 커지게 되면 done=True로 변경
            done = True
        else:
            done = False
        

        return obs, reward, done, {}




    def render(self, render_mode = None):
        # 렌더모드를 휴먼으로 진행시에만 렌더링이 가능하도록 진행
        if render_mode == "human":

            # 여러 보조지표들을 추가하여 그래프에 함께 그리고자 할때,
            # 보조지표가 추가되거나 수정되거나 진행 후 렌더링을 하고자 하면 수정 해야함
            ma_line = go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='MA 20', yaxis='y2')
            upper_band = go.Scatter(x=df['Date'], y=df['Upper'], mode='lines', name='Upper Band', line=dict(width=1),yaxis='y2')
            lower_band = go.Scatter(x=df['Date'], y=df['Lower'], mode='lines', name='Lower Band', line=dict(width=1),yaxis='y2')
            net_worth_line = go.Scatter(x=df['Date'], y=df['Net_Worth'], mode='lines', name='Net Worth',line=dict(color='gold'), yaxis='y')


            # Net worth (예제 계산) -수정 되어야함
            df['Net_Worth'] = df['Close'].cumsum() / 1000  # 예시 계산입니다

            # action 값을 순서대로 따로 저장 해뒀다가 차트그래프와 함께 플랏하는식으로 진행 되어야 할듯. -수정 요망
            df['action'] = np.random.randint(0, 4, df.shape[0])  # 0부터 3까지의 값

            # 캔들스틱
            candle = go.Candlestick(x=df['Date'],open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],increasing_line_color='red', decreasing_line_color='blue',yaxis='y2')


            # 'action' 값에 따른 마커 추가
            # 롱, 숏, 정산 이미지에 대한 내용들
            shapes = []

            # Action 0에 대한 설정
            df_filtered_0 = df[df['action'] == 0]
            marker_0 = go.Scatter(x=df_filtered_0['Date'], y=df_filtered_0['Close'], mode='markers', name='Action 0',marker_symbol='triangle-up', marker_color='red', yaxis='y2')
            shapes.append(marker_0)

            # Action 1에 대한 설정
            df_filtered_1 = df[df['action'] == 1]
            marker_1 = go.Scatter(x=df_filtered_1['Date'], y=df_filtered_1['Close'], mode='markers', name='Action 1',marker_symbol='triangle-down', marker_color='blue', yaxis='y2')
            shapes.append(marker_1)

            # Action 2에 대한 설정
            df_filtered_2 = df[df['action'] == 2]
            marker_2 = go.Scatter(x=df_filtered_2['Date'], y=df_filtered_2['Close'], mode='markers', name='Action 2',marker_symbol='circle', marker_color='green', yaxis='y2')
            shapes.append(marker_2)
            # 모든 플롯 결합 (action=3에 대해서는 아무 것도 추가하지 않음)
            fig = go.Figure(data=[candle, ma_line, upper_band, lower_band, net_worth_line] + shapes)


            # 레이아웃 업데이트
            fig.update_layout(
                xaxis=dict(domain=[0.7, 0.7]),
                yaxis=dict(title='Net Worth', side='right', overlaying='y2'),
                yaxis2=dict(title='Price', side='left'),
                title='Candlestick Chart with Indicators and Secondary Y-Axis for Net Worth',
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
        df = df.drop(['Date'], axis=1)

        # Volume 열에 대해 로그 변환 수행 (거래량의 경우 많은날은 너무 값이 튀어서 문제가 생길 수 있다고 판단되어 로그 스케일로 진행)
        df['Volume'] = np.log(df['Volume'])

        # 로그 변환된 Volume과 다른 열들에 대해 정규화 수행
        normalized_df = (df - df.min()) / (df.max() - df.min())

        return normalized_df



 # df 데이터를 받아 window_size + test_window_size만큼 랜덤 위치로 잘라서 자른 df를 반환해주는 함수
    def generate_random_data_slice(data, window_size, test_window_size):

        max_start_index = len(data) - window_size - test_window_size
        if max_start_index <= 0:
            raise ValueError("데이터 크기가 너무 작아 분할할 수 없습니다.")

        start_index = np.random.randint(0, max_start_index)
        end_index = start_index + window_size
        test_end_index = end_index + test_window_size

        slice_df = data[start_index:test_end_index]
        observation_df = data[start_index:end_index]
        train_df = data[end_index:test_end_index]

        return slice_df,observation_df,train_df


    def next_observation(self):

        observation_df = self.slice_df.iloc[(self.current_index-101)+1:(self.current_index-1)+1]

        return observation_df




window_size = 100  # 에이전트가 볼 수 있는 차트의 크기 (obs 차트 데이터 크기)
test_window_size = 300  # 에이전트가 볼 수 없고 학습을 진행해야 하는 차트의 크기 

df_path = pd.DataFrame # 각자 .csv파일 경로 지정하는 식으로 (ex:"D:\AI_Learning/bitcoin_chart_Data.csv" )



# 테스트용으로 임시적으로 data를 만들어서 진행 해볼 예정
data = [[i + j for j in range(5)] for i in range(1, 1000 + 1)]
temp_test_df =  pd.DataFrame(data, columns=[f"Column_{i+1}" for i in range(5)])
temp_env = stablebaselineEnv(temp_test_df,window_size,test_window_size)
##########################################################################



# 신경망 모델을 만듬 - MultiInputPolicy(stable baseline에서 신경망 구조 알아서 만들어줌), env: 환경을 받아옴, verbose: log print 양 결정(0:전체,1:심플,2:제외)
env = make_vec_env(lambda: stablebaselineEnv(), n_envs=1)

model = PPO("MultiInputPolicy", env, verbose=1) 
obs = env.reset()
