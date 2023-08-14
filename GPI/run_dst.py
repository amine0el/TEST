import numpy as np
import gym
from utils.utils import eval_mo, eval_test_tasks, hypervolume, moving_average, policy_evaluation_mo, random_weights
from tabular_sf import SF
from gpi import GPI
from ols import OLS
import envs
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from utils.utils import get_eval_w
from neptune_logger import NeptuneLogger


def run(algo):

    ccs = [np.array([19.778, -17.383]), np.array([0.7, -1. ]), np.array([11.047, -4.901]), np.array([8.037, -2.97 ]), np.array([14.856, -8.648]), np.array([ 17.373, -12.248]), np.array([ 19.073, -15.706]), np.array([13.181, -6.793]), np.array([14.074, -7.726]), np.array([17.814, -13.125])] 
    ccs_weights = [np.array([1., 0.]), np.array([0., 1.]), np.array([0.462, 0.538]), np.array([0.274, 0.726]), np.array([0.588, 0.412]), np.array([0.64, 0.36]), np.array([0.681, 0.319]), np.array([0.498, 0.502]), np.array([0.536, 0.464]), np.array([0.67, 0.33])]
    #ccs_weights = np.array(get_eval_w(32,2))
    env = gym.make("DeepSeaTreasure-v0")
    eval_env = gym.make("DeepSeaTreasure-v0")
    nl = NeptuneLogger()
    nl.start(run_name='DST-SFOLS', tags=['DST-SFOLS','DST'])

    agent_constructor = lambda: SF(env,
                                alpha=0.3,
                                gamma=0.99,
                                initial_epsilon=1.0,
                                final_epsilon=0.05,
                                use_replay=True,
                                per=True,
                                use_gpi=True,
                                envelope=False,
                                batch_size=5,
                                buffer_size=500000,
                                epsilon_decay_steps=100000,
                                project_name='DST-SFOLS',
                                log=True,
                                logger=nl,)
    gpi_agent = GPI(env,
                    agent_constructor,
                    log=True,
                    logger= nl,
                    project_name='DST-SFOLS',
                    experiment_name=algo)

    ols = OLS(m=2, epsilon=0.01)
    test_tasks = random_weights(dim=2, seed=42, n=64) + ccs_weights
    max_iter = 10
    for iter in range(max_iter):
        if algo == 'SFOLS':
            w = ols.next_w()
        elif algo == 'WCPI':
            w = ols.worst_case_weight()
        elif algo == 'Random':
            w = random_weights(dim=2)
        print('next w', w)
        total_timesteps=100000
        gpi_agent.learn(total_timesteps=total_timesteps,
                    use_gpi=True,
                    w=w,
                    eval_env=eval_env,
                    eval_freq=1000,
                    reset_num_timesteps=False,
                    new_policy=True,
                    reset_learning_starts=False)
        

        value = policy_evaluation_mo(gpi_agent, eval_env, w, rep=5)
        remove_policies = ols.add_solution(value, w, gpi_agent=gpi_agent, env=eval_env)       
        gpi_agent.delete_policies(remove_policies)

        #ols.plot_ccs(ccs, ccs_weights, gpi_agent, eval_env)

        gpi_agent.get_metric(total_timesteps*(iter+1))
        returns = [policy_evaluation_mo(gpi_agent, eval_env, w, rep=5, return_scalarized_value=False) for w in test_tasks]
        print('returns', returns)
        returns_ccs = [policy_evaluation_mo(gpi_agent, eval_env, w, rep=5, return_scalarized_value=False) for w in ols.ccs_weights]
        mean_test = np.mean([np.dot(psi, w) for (psi, w) in zip(returns, test_tasks)], axis=0)
        nl.log_metric(metric_value=mean_test, metric_name='mean_value_test_tasks', step=ols.iteration, mode='metrics')
        mean_test_smp = np.mean([ols.max_scalarized_value(w_test) for w_test in test_tasks]) 
        nl.log_metric(metric_value=hypervolume(np.array([0.0, -17.383]), ols.ccs), metric_name='hypervolume_o', step=ols.iteration, mode='metrics')
        nl.log_metric(metric_value=hypervolume(np.array([0.0, -17.383]), returns+returns_ccs), metric_name='hypervolume_GPI', step=ols.iteration, mode='metrics')
        nl.log_metric(metric_value=mean_test_smp, metric_name='mean_value_test_tasks_SMP', step=ols.iteration, mode='metrics')
        if ols.ended():
            print("ended at iteration", iter)
            for i in range(ols.iteration + 1, max_iter + 1):
                nl.log_metric(metric_value=mean_test, metric_name='mean_value_test_tasks', step=i, mode='metrics')
                nl.log_metric(metric_value=mean_test_smp, metric_name='mean_value_test_tasks_SMP', step=i, mode='metrics')
                nl.log_metric(metric_value=hypervolume(np.array([0.0, -17.383]), ols.ccs), metric_name='hypervolume', step=i, mode='metrics')
                nl.log_metric(metric_value=hypervolume(np.array([0.0, -17.383]), returns+returns_ccs), metric_name='hypervolume_GPI', step=i, mode='metrics')
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Sea Treasure experiment.')
    parser.add_argument('-algo', type=str, choices=['SFOLS', 'WCPI', 'Random'], default='SFOLS', help='Algorithm.')
    args = parser.parse_args()

    run(args.algo)
