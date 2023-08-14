import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics

from pcn import PCN


def main():
    def make_env():
        env = mo_gym.make("deep-sea-treasure-v0")
        env = MORecordEpisodeStatistics(env, gamma=1.0)
        return env

    env = make_env()

    agent = PCN(
        env,
        scaling_factor=np.array([1, 1, 0.1]),
        learning_rate=1e-3,
        batch_size=256,
        project_name="MORL-Baselines",
        log=True,
    )

    agent.train(
        eval_env=make_env(),
        total_timesteps=int(100000),
        ref_point=np.array([0, -25]),
        num_er_episodes=200,
        max_buffer_size=500,
        num_model_updates=500,
        max_return=np.array([23.7, -1]),
        known_pareto_front=env.unwrapped.pareto_front(gamma=1.0),
    )


if __name__ == "__main__":
    main()
