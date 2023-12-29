"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""
from __future__ import annotations

import glob
import os
import time

import supersuit as ss
import torch
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import get_device
from gymnasium.utils import EzPickle
from pettingzoo.utils import parallel_to_aec

from envs.ma_quadx_hover_env import MAQuadXHoverEnv


def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, seed: int | None = 0, train_desc = '', resume = False, **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**env_kwargs)

    env.reset(seed=seed)

    env = ss.black_death_v3(env)
    #print(f'{env.black_death=}')

    env = ss.pettingzoo_env_to_vec_env_v1(env,)
    env.black_death = True


    num_vec_envs = 1 #8
    num_cpus = 1 #(os.cpu_count() or 1)
    env = ss.concat_vec_envs_v1(env, num_vec_envs, num_cpus=num_cpus, base_class="stable_baselines3", )

    device = get_device('cuda')
    batch_size = 512
    lr = 1e-3
    nn_t = [256, 256, 256]
    policy_kwargs = dict(
        net_arch=dict(pi=nn_t, vf=nn_t)
    )


    model = PPO(
    MlpPolicy,
    env,
    verbose=1,
    learning_rate=lr,
    batch_size=batch_size,
    policy_kwargs=policy_kwargs,
    device=device,
    )
    model_name = f"models/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    model.learn(total_timesteps=steps)
    model.save(model_name)

    print("Model has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")


def eval(env_fn, num_games: int = 100, render_mode: str | None = None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode, **env_kwargs )
    env = parallel_to_aec(env)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    try:
        latest_policy = max(
            glob.glob(f"models/{env.metadata['name']}*.zip"), key=os.path.getctime
        )
    except ValueError:
        print("Policy not found.")
        exit(0)
    print(f"Using {latest_policy} as model.")
    model = PPO.load(latest_policy)

    rewards = {agent: 0.0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # SB3 backup are designed for single-agent settings, we get around this by using he same model for every agent

    for i in range(num_games):
        last_term = False
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if truncation : #and info.get('mission_complete') == True
                print(f'terminate with {agent=} {termination=} {truncation=} {info=}')
                break
            else:
                act = model.predict(obs, deterministic=True)[0]
                #act = np.array([1,1,0,0])
            if termination != last_term:
                print(f'| A agent terminated |')
                print(f'{obs=}')
                print(f'{agent=}')
                print(f'{termination=}')
                print(f'{truncation=}\n')
                print(f'{reward=}\n')
                print(f'{info}')
            env.step(act)
            #print(f'{reward=}')



    env.close()

    avg_reward_per_agent = sum(rewards.values()) / len(rewards.values()  )
    avg_reward_per_game = sum(rewards.values()) / num_games
    print("\nRewards: ", rewards)
    print(f"Avg reward per agent: {avg_reward_per_agent}")
    print(f"Avg reward per game: {avg_reward_per_game}")
    return avg_reward_per_game


class EZPEnv(EzPickle, MAQuadXHoverEnv):
    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        MAQuadXHoverEnv.__init__(self, *args, **kwargs)


seed=None

spawn_settings = dict(
    lw_center_bounds=10.0,
    lm_center_bounds=10.0,
    lw_spawn_radius=4.0,
    lm_spawn_radius=10,
    min_z=1.0,
    seed=None,
    num_lw=6,
    num_lm=12,
)

if __name__ == "__main__":
    env_fn = EZPEnv

    env_kwargs = {}


    #Train a model (takes ~3 minutes on GPU)
    train_butterfly_supersuit(env_fn, steps=10_000_000, **env_kwargs)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    #eval(env_fn, num_games=10, render_mode=None, **env_kwargs)

    #eval(env_fn, num_games=1, render_mode="human", **env_kwargs)
