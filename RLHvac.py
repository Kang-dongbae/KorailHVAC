import numpy as np
import pandas as pd
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# file load
data = pd.read_csv(r'C:\Dev\KorailHVAC\filtered_data_sorted.csv', encoding='euc-kr')
# print(data.head())

def drop_columns(data):
    columns = [
    '000 HVAC SDR_T CAR 외부온도',
    '000 HVAC SDR_T CAR 온도 설정치 변동폭',
    '100 HVAC SDR_M CAR 외부온도',
    '100 HVAC SDR_M CAR 온도 설정치 변동폭',
    '200 HVAC SDR_T CAR 외부온도',
    '200 HVAC SDR_T CAR 온도 설정치 변동폭',
    '300 HVAC SDR_T CAR 외부온도',
    '300 HVAC SDR_T CAR 온도 설정치 변동폭',
    '400 HVAC SDR_T CAR 외부온도',
    '400 HVAC SDR_T CAR 온도 설정치 변동폭',
    '500 HVAC SDR_T CAR 외부온도',
    '500 HVAC SDR_T CAR 온도 설정치 변동폭',
    '600 HVAC SDR_T CAR 외부온도',
    '600 HVAC SDR_T CAR 온도 설정치 변동폭',
    '700 HVAC SDR_T CAR 외부온도',
    '700 HVAC SDR_T CAR 온도 설정치 변동폭',
    '800 HVAC SDR_T CAR 외부온도',
    '800 HVAC SDR_T CAR 온도 설정치 변동폭',
    '900 HVAC SDR_T CAR 외부온도',
    '900 HVAC SDR_T CAR 온도 설정치 변동폭'
    ]
    return data.drop(columns=columns)

data = drop_columns(data)
print(data.head())


'''
# DQN 신경망 정의
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 경험 재생을 위한 리플레이 메모리
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# HVAC 제어 환경 시뮬레이터
class HVACEnv:
    def __init__(self, data):
        self.data = data
        self.current_idx = 0
        self.n = len(data)
        self.action_space = [0, 1, 2]  # 0: 미가동, 1: 반냉방, 2: 전냉방
        self.CO2_threshold = 2000
        self.T_set = 24.0  # 설정 온도
        self.P_max = 5.0   # 최대 소비 전력(kW)
    
    def reset(self):
        self.current_idx = 0
        return self._get_state(self.current_idx)
    
    def _get_state(self, idx):
        # 상태 벡터: [Tin_avg, Tout_avg, CO2_avg, ΔTin, ΔTout, ΔCO2, Δt]
        if idx == 0:
            prev = self.data.iloc[idx]
        else:
            prev = self.data.iloc[idx-1]
        
        curr = self.data.iloc[idx]
        state = np.array([
            curr['Tin_mean'],
            curr['Tout_mean'],
            curr['CO2_mean'],
            curr['Tin_mean'] - prev['Tin_mean'],
            curr['Tout_mean'] - prev['Tout_mean'],
            curr['CO2_mean'] - prev['CO2_mean'],
            curr['duration']
        ], dtype=np.float32)
        return state
    
    def step(self, action):
        curr = self.data.iloc[self.current_idx]
        Tin = curr['Tin_mean']
        CO2 = curr['CO2_mean']
        
        # 액션에 따른 전력 소비 계산
        if action == 0:    # 미가동
            P_HVAC = 0
        elif action == 1:  # 반냉방
            P_HVAC = self.P_max * 0.5
        elif action == 2:  # 전냉방
            P_HVAC = self.P_max
        
        # 보상 구성 요소 계산
        delta_T = abs(Tin - self.T_set)
        R_comfort = -delta_T / 5.0  # 편의성 보상
        
        R_energy = -P_HVAC / self.P_max  # 에너지 보상
        
        # 공기질 보상 (CO2 > 2000ppm 시 패널티)
        R_air_quality = -max(0, CO2 - 2000) / 3000
        
        # 가중치 적용된 총 보상
        w1, w2, w3 = 0.4, 0.4, 0.2  # 가중치 (편의성, 에너지, 공기질)
        reward = w1 * R_comfort + w2 * R_energy + w3 * R_air_quality
        
        # 다음 상태 업데이트
        self.current_idx += 1
        done = self.current_idx >= self.n - 1
        next_state = self._get_state(self.current_idx) if not done else None
        
        return next_state, reward, done

# DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99    # 할인 계수
        self.epsilon = 1.0    # 탐험률
        self.epsilon_min = 0.01
        self.epsilon_decay
'''