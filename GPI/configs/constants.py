PROJECT = 'MORLBENCHMARK/MORLBENCH'
TMP_STORAGE = './.neptune/neptune_tmp_storage'

ON_POLICY_ALGOS = ['ac', 'va']
OFF_POLICY_ALGOS = ['dqn', 'sac', 'pgmorl']
PQL_ALGOS = ['pql']
PCN_ALGOS = ['pcn']
STATELESS_ALGOS = ['resnet']
EPSILON_GREEDY_ALGOS = ['dqn']
STATELESS_ENVS = ['fabric-defect-detection', 'generalized-cls']
SEQUENTIAL_ENVS = ['so_minecart', 'deep-sea-treasure', 'halfcheetah', 'fishwood', 'water-reservoir', 'so_pole',
                   'so_pendulum', 'highway', 'minecart']