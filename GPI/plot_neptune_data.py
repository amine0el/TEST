import os

import torch
import mo_gymnasium as mo_gym
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import calc_opt_reward

from neptune_logger import NeptuneLogger


def get_group_name(groups, run_name):
    """

    Parameters
    ----------
    groups
    run_name

    Returns
    -------

    """
    for key, value in groups.items():
        if run_name in value:
            return key


def gen_plot(df_all, plot_config, figsize):
    """

    Parameters
    ----------
    df_all
    plot_config
    """
    for metric in plot_config['metrics']:
        fig, ax = plt.subplots(figsize=figsize)
        df_filt = df_all[
            (df_all.metric == metric) & (df_all[plot_config['grouping_name']].isin(plot_config['exps']))]
        if metric == 'pref_loss':
            metric = 'EUL'
        else:
            metric = metric.capitalize()
        df_filt.rename(columns={'value': metric}, inplace=True)
        df_filt.Algorithm = df_filt.Algorithm.str.replace(f'-{plot_config["u_func_str"].capitalize()}', '')
        sns.lineplot(data=df_filt,
                     x=plot_config['x_name'],
                     y=metric,
                     hue=plot_config['grouping_name'],
                     ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_config['save_path'], f'training_{plot_config["u_func_str"]}_{metric}.pdf'))


# def gen_precision_plot(df_results, plot_config, opt_reward):
#     color1 = 'tab:orange'
#     color2 = 'tab:blue'
#     for exp in plot_config['exps']:
#         df_filt = df_results[df_results['exp'] == exp]
#         fig, ax1 = plt.subplots(figsize=plot_config['figsize'])
#         sns.lineplot(data=df_filt, x='pref_0', y='reward_0', ax=ax1, legend=False, color=color1)
#         ax1.set_ylabel(f'Reward Objective 0', color=color1)
#         ax2 = ax1.twinx()
#         sns.lineplot(data=df_filt, x='pref_0', y='reward_1', ax=ax2, legend=False, color=color2)
#         ax2.set_ylabel(f'Reward Objective 1', color=color2)
#         ax1.plot(prefs[:, 0], opt_reward[:, 0], color=color1, linestyle='--')
#         ax2.plot(prefs[:, 0], opt_reward[:, 1], color='black', linestyle='--', label='Optimal Rewards')
#         ax2.plot(prefs[:, 0], opt_reward[:, 1], color=color2, linestyle='--')
#         ax1.set_title(exp.replace('-Linear', '').replace('-Square', '').replace('-Log', ''))
#         ax1.set_xlabel('Preference for Objective 0')
#         plt.legend(loc='lower center')
#         plt.tight_layout()
#         plt.savefig(os.path.join(plot_config['save_path'], f'precision_{exp}.pdf'))


def fetch_data_from_neptune(groups, base_path, metrics, result_file):
    nl = NeptuneLogger()
    for group, runs in groups.items():
        folder_path = os.path.join(base_path, group)
        os.makedirs(folder_path, exist_ok=True)
        for run in runs:
            run_path = os.path.join(folder_path, run)
            os.makedirs(run_path, exist_ok=True)
            nl.start('', None, f'OP-{run}')
            nl.download_artifact('csvs', name=result_file, save_path=run_path)
            for metric in metrics:
                nl.download_metrics('metrics', name=metric, save_path=run_path)
            nl.stop()


def get_opt_reward(eval_prefs, u_func):
    env = mo_gym.make('deep-sea-treasure-v0')
    treasures = env.sea_map > 0
    value = env.sea_map[treasures]
    cost = -sum(np.where(treasures))
    pareto_frontier = torch.tensor(np.array([value, cost]).T)
    return calc_opt_reward(torch.tensor(eval_prefs), pareto_frontier, u_func)


def load_data(config, base_path):
    df_results = pd.DataFrame()
    df_all = pd.DataFrame()
    for exp in config['exps']:
        exp_path = os.path.join(base_path, exp)
        if exp[0] == '.':
            continue
        for run in os.listdir(exp_path):
            if run[0] == '.':
                continue
            run_path = os.path.join(exp_path, run)
            for metric in config['metrics']:
                if metric[0] == '.':
                    continue
                file_path = os.path.join(run_path, f'{metric}.csv')
                df = pd.read_csv(file_path, index_col=0)
                df['run'] = run
                df['metric'] = metric
                df[config['grouping_name']] = exp
                df_all = pd.concat([df_all, df])
            file_path = os.path.join(run_path, config['result_file'])
            df = pd.read_csv(file_path, index_col=0)
            df['run'] = run
            df['exp'] = exp
            df_results = pd.concat([df_results, df])
    df_all.rename(columns={"step": config['x_name']}, inplace=True)
    return df_results, df_all
