import numpy as np

from evaluation import eval_mo
from pgmorl import PGMORL
from mo_ppo import make_env


if __name__ == "__main__":
    env_id = "mo-halfcheetah-v4"
    env = make_env(env_id, 422, 1, "PGMORL_test", gamma=0.995)()
    algo = PGMORL(
        env_id=env_id,
        origin=np.array([1.0, 1.0]),
        num_envs=4,
        pop_size=6,
        warmup_iterations=8,
        evolutionary_iterations=2,
        num_weight_candidates=7,
    )
    algo.train(
        eval_env=env,
        total_timesteps=int(5e6),
        ref_point=np.array([0.0, -5.0]),
        known_pareto_front=None,
    )
    env = make_env(env_id, 422, 1, "PGMORL_test", gamma=0.995)()  # idx != 0 to avoid taking videos

    # Execution of trained policies
    for a in algo.archive.individuals:
        scalarized, discounted_scalarized, reward, discounted_reward = eval_mo(
            agent=a, env=env, w=np.array([1.0, 1.0]), render=True
        )
        print(f"Agent #{a.id}")
        print(f"Scalarized: {scalarized}")
        print(f"Discounted scalarized: {discounted_scalarized}")
        print(f"Vectorial: {reward}")
        print(f"Discounted vectorial: {discounted_reward}")
