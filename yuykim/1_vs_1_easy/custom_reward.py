import numpy as np
import gym

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_ball_owned = False
        # 1대1이므로 상대 수비수 1명을 제쳤는지 여부만 체크
        self.is_defender_passed = False 

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs_dict = self.obs_to_dict(obs)
        # 초기화
        self.prev_ball_owned = (obs_dict["ball_ownership"][1] == 1)
        self.is_defender_passed = False
        return obs

    def step(self, action):
        # 1. 환경 실행 (기본 reward: 득점 시 +1, 실점 시 -1)
        obs, reward, done, info = self.env.step(action)
        obs_dict = self.obs_to_dict(obs)
        
        # 현재 상태 정보 추출
        current_ball_owned = (obs_dict["ball_ownership"][1] == 1)
        opp_ball_owned = (obs_dict["ball_ownership"][2] == 1)
        
        # 조종 중인 우리 선수 위치 (x, y)
        active_idx = np.argmax(obs_dict["active_player"])
        my_pos = obs_dict["left_team"][active_idx*2 : active_idx*2+2]
        
        # 1대1 상황이므로 상대팀의 첫 번째 선수(보통 수비수) 위치만 파악
        opp_pos = obs_dict["right_team"][0:2]

        # --- 보상 설계 ---

        # [케이스 1] 슈팅 시도 (Action 12)
        # 골대 근처에서 슛을 쏘는 습관을 기르기 위해 보상 부여
        if action == 12 and current_ball_owned:
            reward += 0.2
            print(">> 슈팅! (+0.2)")

        # [케이스 2] 수비 성공 (상대가 가진 공을 뺏어옴)
        if not self.prev_ball_owned and current_ball_owned:
            reward += 1.0
            print(">> 수비 성공! (+1.0)")

        # [케이스 3] 상대 제치기
        # 수비수보다 x좌표가 커지고(앞지름), 거리가 어느 정도 가까울 때
        dist = np.linalg.norm(my_pos - opp_pos)
        if not self.is_defender_passed and current_ball_owned and my_pos[0] > opp_pos[0] and dist < 0.2:
            reward += 0.8
            self.is_defender_passed = True # 한 판에 한 번만 부여
            print(">> 수비수 돌파! (+0.8)")

        # 추가 패널티: 공을 뺏겼을 때
        if self.prev_ball_owned and opp_ball_owned:
            reward -= 0.5
            print(">> 공 뺏김! (-0.5)")

        # 상태 업데이트
        self.prev_ball_owned = current_ball_owned
        
        return obs, reward, done, info

    def obs_to_dict(self, obs):
        # 22 - (x,y) coordinates of left team players
        # 22 - (x,y) direction of left team players
        # 22 - (x,y) coordinates of right team players
        # 22 - (x, y) direction of right team players
        # 3 - (x, y and z) - ball position
        # 3 - ball direction
        # 3 - one hot encoding of ball ownership (noone, left, right)
        # 11 - one hot encoding of which player is active
        # 7 - one hot encoding of `game_mode`

        obs_dict = {}
        obs_dict["left_team"] = obs[0:22]
        obs_dict["left_team_direction"] = obs[22:44]
        obs_dict["right_team"] = obs[44:66]
        obs_dict["right_team_direction"] = obs[66:88]
        obs_dict["ball"] = obs[88:91]
        obs_dict["ball_direction"] = obs[91:94]
        obs_dict["ball_ownership"] = obs[94:97]
        obs_dict["active_player"] = obs[97:108]
        obs_dict["game_mode"] = obs[108:115]

        return obs_dict
