import numpy as np
import gym

class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.waiting_for_receiver = False
        self.kicker_index = -1
        self.prev_ball_owned = False
        self.prev_game_mode = 0 # 0: Normal

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs_dict = self.obs_to_dict(obs)
        self.waiting_for_receiver = False
        self.kicker_index = -1
        self.prev_ball_owned = (obs_dict["ball_ownership"][1] == 1)
        self.prev_game_mode = np.argmax(obs_dict["game_mode"])
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_dict = self.obs_to_dict(obs)
        
        # 현재 상태 추출
        current_ball_owned_team = np.argmax(obs_dict["ball_ownership"]) # 0:없음, 1:우리, 2:상대
        current_ball_owned = (current_ball_owned_team == 1)
        current_game_mode = np.argmax(obs_dict["game_mode"])
        active_player_idx = np.argmax(obs_dict["active_player"])

        # --- 보상 로직 ---

        # [케이스 1] 롱킥(Action 10) 성공 보상 (+0.2)
        if action == 10 and current_ball_owned:
            self.waiting_for_receiver = True
            self.kicker_index = active_player_idx

        if self.waiting_for_receiver:
            if current_ball_owned and active_player_idx != self.kicker_index:
                reward += 0.2
                self.waiting_for_receiver = False
                print(f">> 롱킥 전달 성공! (+0.2)")
            elif current_ball_owned_team == 2 or done:
                self.waiting_for_receiver = False

        # [케이스 2] 라인 아웃 및 소유권 변경 감점 (-0.3)
        # 조건: 인플레이(Normal) 상황이었다가 세트피스 상황으로 변했는데, 소유권이 상대에게 있을 때
        if self.prev_game_mode == 0 and current_game_mode != 0:
            if current_ball_owned_team == 2: # 상대방 공이 됨 (드로인, 골킥 등)
                reward -= 0.3
                print(f">> 라인 아웃! 상대에게 소유권 넘어감 (-0.3) | 모드: {current_game_mode}")

        # [케이스 3] 기본 득점 보상
        if reward > 0.9:
            print(">> 골!!! (+1.0)")

        # --- 상태 업데이트 ---
        self.prev_ball_owned = current_ball_owned
        self.prev_game_mode = current_game_mode
        
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
