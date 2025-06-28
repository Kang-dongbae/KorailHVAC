import numpy as np
import pandas as pd
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# file load
#data = pd.read_csv(r'C:\Dev\KorailHVAC\filtered_data_sorted.csv', encoding='euc-kr')
data = pd.read_csv(r'filtered_data_sorted.csv', encoding='euc-kr')
#print(data.head())

data['발생시각'] = pd.to_datetime(data['발생시각'], errors='coerce')

def process_car_data(data):
     # 필요한 컬럼 정의
    required_columns = [
        '발생시각',
        'HVAC SD_실내온도 값',
        'HVAC SD_실외온도 값',
        'HVAC SD_CO2 센서 값',
        'HVAC SD_전냉 CFULL',
        'HVAC SD_반냉 CHALF',
        'HVAC SD_OFF',
        'HVAC SDR_T CAR 자동 AUTO'
    ]
    
    # 변경할 컬럼 이름 매핑
    column_rename = {
        '발생시각': 'Time',
        'HVAC SD_실내온도 값': 'InTemp',
        'HVAC SD_실외온도 값': 'OutTemp',
        'HVAC SD_CO2 센서 값': 'CO2',
        'HVAC SD_전냉 CFULL': 'CFULL',
        'HVAC SD_반냉 CHALF': 'CHALF',
        'HVAC SD_OFF': 'OFF',
        'HVAC SDR_T CAR 자동 AUTO': 'Auto'
    }
    
    # 컬럼 삭제
    columns_to_drop = [f"{i:03d} HVAC SDR_T CAR {'외부온도' if j == 0 else '온도 설정치 변동폭'}"
                       for i in range(0, 1000, 100) for j in range(2)]
    data = data.drop(columns=columns_to_drop, errors='ignore')

    # 컬럼 이름 변경 (DEFAULT_DD{i} → Door_{Left/Right})
    old_columns = [f'DEFAULT_DD{i}' for i in range(153, 230, 4)]
    new_columns = [f'{i:03d} Door_{"Left" if j == 0 else "Right"}'
                   for i in range(0, 1000, 100) for j in range(2)]
    data = data.rename(columns=dict(zip(old_columns, new_columns)))

    # 공통 컬럼 정의
    common_columns = {
        '발생시각': '발생시각',
        'DEFAULT_HCR': 'HCR',
        'DEFAULT_TCR': 'TCR',
        'DEFAULT_현재역': '현재역',
        'DEFAULT_다음역': '다음역',
        'DEFAULT_종착역': '종착역',
        'DEFAULT_신호장치 속도': '신호장치 속도'
    }

    # 컬럼 매핑 생성
    column_mapping = {
        col: (None, common_columns[col]) if col in common_columns else
             (col[:3], col[4:]) if col.startswith(tuple(f"{i:03d} " for i in range(0, 1000, 100)))
             else (None, col)
        for col in data.columns
    }

    # 호차별 데이터프레임 생성
    car_dfs = {}
    for car_num in [f"{i:03d}" for i in range(0, 1000, 100)]:
        car_columns = [col for col, (num, _) in column_mapping.items() if num == car_num or num is None]
        temp_df = data[car_columns].copy()
        temp_df.columns = [common_columns.get(col, column_mapping[col][1]) for col in car_columns]
        
        # 필요한 컬럼만 필터링
        available_columns = [col for col in required_columns if col in temp_df.columns]
        if available_columns:
            temp_df = temp_df[available_columns]
            # 컬럼 이름 변경
            temp_df = temp_df.rename(columns=column_rename)
            # 결측값 처리
            #temp_df['Time'] = pd.to_datetime(temp_df['Time'], errors='coerce')
            #temp_df['InTemp'] = temp_df['InTemp'].fillna(25.0)
            #temp_df['OutTemp'] = temp_df['OutTemp'].fillna(15.0)
            #temp_df['CO2'] = temp_df['CO2'].fillna(500.0)
            #temp_df['CFULL'] = temp_df['CFULL'].fillna(0)
            #temp_df['CHALF'] = temp_df['CHALF'].fillna(0)
            #temp_df['OFF'] = temp_df['OFF'].fillna(0)
            # 중복 제거
            car_dfs[f"{int(car_num)//100 + 1}호차"] = temp_df

    print("completed process_car_data")
    return car_dfs

# 데이터 전처리

car_dfs = process_car_data(data)
print(car_dfs['1호차'].head())
car_dfs['1호차'].head(3000).to_csv('processed_1.csv', index=False, encoding='utf-8-sig')

'''
# HVAC 제어를 위한 DQN 에이전트 구현
# DQN 신경망 정의
class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 환경 클래스 정의
class HVACEnvironment:
    def __init__(self, car_df, target_temp=23, max_co2=800):
        self.df = car_df.reset_index(drop=True)
        self.target_temp = target_temp
        self.max_co2 = max_co2
        self.current_step = 0
        self.actions = ['Full_AC', 'Half_AC', 'Off']  # 전냉방, 반냉방, 미가동
        self.state_size = 4  # 실내온도, 실외온도, CO2, 시간대
        self.action_size = len(self.actions)
        
    def reset(self):
        self.current_step = 0
        return self._get_state()
    
    def _get_state(self):
        if self.current_step >= len(self.df):
            return np.zeros(self.state_size, dtype=np.float32)
        row = self.df.iloc[self.current_step]
        indoor_temp = row['HVAC SD_실내온도 값'] if pd.notna(row['HVAC SD_실내온도 값']) else 25.0
        outdoor_temp = row['HVAC SD_실외온도 값'] if pd.notna(row['HVAC SD_실외온도 값']) else 15.0
        co2 = row['HVAC SD_CO2 센서 값'] if pd.notna(row['HVAC SD_CO2 센서 값']) else 500.0
        hour = pd.to_datetime(row['발생시각']).hour
        return np.array([indoor_temp, outdoor_temp, co2, hour], dtype=np.float32)

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_state(), 0, True
        
        # 현재 상태
        row = self.df.iloc[self.current_step]
        indoor_temp = row['HVAC SD_실내온도 값'] if pd.notna(row['HVAC SD_실내온도 값']) else 25.0
        co2 = row['HVAC SD_CO2 센서 값'] if pd.notna(row['HVAC SD_CO2 센서 값']) else 500.0
        
        # 행동에 따른 온도 변화 (간단한 시뮬레이션)
        if action == 0:  # 전냉방
            indoor_temp -= 2.0
            energy_cost = 5
        elif action == 1:  # 반냉방
            indoor_temp -= 1.0
            energy_cost = 2
        else:  # 미가동
            indoor_temp += 0.5
            energy_cost = 0
        
        # 보상 계산
        temp_diff = abs(indoor_temp - self.target_temp)
        co2_penalty = 0.01 * max(0, co2 - self.max_co2)
        reward = 10 - temp_diff - co2_penalty - energy_cost
        
        # 다음 단계로 이동
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_state = self._get_state()
        
        return next_state, reward, done

# DQN 에이전트
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 할인율
        self.epsilon = 1.0  # 탐험 비율
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.update_target_model()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# DQN 에이전트 학습
def train_dqn(car_dfs, episodes=50, batch_size=32):
    for car_name, df in car_dfs.items():
        print(f"Training DQN for {car_name}")
        df['발생시각'] = pd.to_datetime(df['발생시각'], errors='coerce')
        env = HVACEnvironment(df)
        agent = DQNAgent(env.state_size, env.action_size)
        
        for e in range(episodes):
            state = env.reset()
            total_reward = 0
            for _ in range(len(df)):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                agent.replay(batch_size)
                if done:
                    break
            agent.update_target_model()
            print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
        
        # 학습된 모델 저장
        torch.save(agent.model.state_dict(), f"{car_name}_dqn_model.pth")

train_dqn(car_dfs)

'''