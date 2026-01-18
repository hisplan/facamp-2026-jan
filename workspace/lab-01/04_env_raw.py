from pprint import pprint
import gfootball.env as football_env

# raw representation
env = football_env.create_environment(
    env_name="11_vs_11_competition", representation="raw", render=False
)

obs = env.reset()

print("gfootball eenvironment reset OK")
print("action space:", env.action_space)

obs = obs[0]

# ref: https://github.com/google-research/football/blob/master/gfootball/doc/observation.md#raw-observations
for key in obs.keys():
    print(key)

pprint(obs)

env.close()
