import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 사용자 정의 환경 클래스 정의, gym.Env를 상속받습니다.
class CustomEnv(gym.Env):
    """OpenAI Gym을 위한 사용자 정의 환경."""
    metadata = {'render.modes': ['console']}  # 렌더링 모드 설정, 여기서는 콘솔 출력만 지원

    def __init__(self):
        super(CustomEnv, self).__init__()  # 부모 클래스의 생성자 호출
        # 행동 공간과 관측 공간 정의
        self.action_space = spaces.Discrete(2)  # 행동 공간 정의: 2개의 이산적인 액션 (0: 왼쪽으로 이동, 1: 오른쪽으로 이동)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)  # 관측 공간 정의: 0부터 100까지의 연속적인 값

        # 초기 상태 설정
        self.state = 50 + self.observation_space.sample()  # 에이전트의 초기 위치 설정
        self.goal = 100  # 목표 상태 정의

    def step(self, action):
        # 에이전트의 행동에 따른 상태 업데이트
        if action == 0:
            self.state -= 1
        else:
            self.state += 1

        # 목표에 도달했는지 여부 확인
        done = bool(self.state >= self.goal)

        # 보상 정의: 목표에 도달했다면 1, 아니면 0
        if done:
            reward = 1.0
        else:
            reward = 0.0

        return np.array([self.state]).astype(np.float32), reward, done, {}

    def reset(self):
        # 환경을 초기 상태로 리셋
        self.state = 50 + self.observation_space.sample()
        return np.array([self.state]).astype(np.float32)

    def render(self, mode='console'):
        # 환경의 현재 상태를 출력
        if mode != 'console':
            raise NotImplementedError()
        print(f'State: {self.state}')


# 사용자 정의 환경을 병렬로 실행할 수 있게 벡터화
env = make_vec_env(lambda: CustomEnv(), n_envs=1)

# PPO 모델 생성 및 학습을 위한 설정
model = PPO('MlpPolicy', env, verbose=1)  # 'MlpPolicy': 다층 퍼셉트론(MLP)을 사용한 정책, env: 학습할 환경, verbose: 학습 과정 로깅
model.learn(total_timesteps=10000)  # total_timesteps: 총 학습 스텝 수

# 학습된 모델로 환경 테스트
obs = env.reset()  # 환경을 초기 상태로 리셋하고 초기 관측값을 받음
for i in range(1000):  # 1000 스텝 동안 에이전트 실행
    action, _states = model.predict(obs, deterministic=True)  # 학습된 모델을 사용해 현재 관측에서의 액션 예측
    obs, rewards, dones, info = env.step(action)  # 예측된 액션을 환경에 적용
    env.render()  # 환경의 현재 상태 출력