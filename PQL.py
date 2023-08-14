import numpy as np
import mo_gymnasium as mo_gym

def dominates(row, candidateRow, evalfunction):
    return sum([row[x]<candidateRow[x] for x in range(evalfunction)])==evalfunction

def simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row, len(candidateRow)):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow, len(row)):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

env = mo_gym.make("deep-sea-treasure-v0")

n_actions = env.action_space.n
state_space_range = int(np.prod(env.observation_space.high - env.observation_space.low + 1))
state_dim = env.observation_space.shape[0]

Q_set = [[[] for _ in range(n_actions)] for _ in range(state_space_range)]
R = np.zeros((state_space_range, n_actions, env.reward_space.shape[0]))
n_sa = np.zeros((state_space_range, n_actions))

EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 500  # adjust this to control the rate of decay

def get_epsilon(t):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * t / EPS_DECAY)

def get_state_index(state):
    state -= env.observation_space.low
    state_index = 0
    for i in range(state_dim):
        state_index *= state_space_range // (env.observation_space.high[i] - env.observation_space.low[i] + 1)
        state_index += state[i]
    return int(state_index)

def choose_action(state, epsilon):
    state_index = get_state_index(state)
    if np.random.random() < epsilon:
        return np.random.choice(n_actions)
    else:
        Q_values = [np.mean([np.sum(solution) for solution in Q_set[state_index][action]]) if len(Q_set[state_index][action]) > 0 else 0 for action in range(n_actions)]
        return np.argmax(Q_values)

for episode in range(10000):
    state = env.reset()[0]
    done = False
    absorbing = False
    print(episode)
    while not (done or absorbing):
        epsilon = get_epsilon(episode)
        action = choose_action(state, epsilon)
        next_state, reward, done, absorbing, _ = env.step(action)
        
        state_index = get_state_index(state)
        next_state_index = get_state_index(next_state)
        
        n_sa[state_index][action] += 1
        R[state_index][action] += (reward - R[state_index][action]) / n_sa[state_index][action]
        
        for a in range(n_actions):
            Q_set_list = list(Q_set[next_state_index][a]) + [R[next_state_index][a]]
            Q_set[next_state_index][a] = simple_cull(Q_set_list, dominates)[0]

        state = next_state

env = mo_gym.make("deep-sea-treasure-v0", render_mode="human")
# Reset the environment
state = env.reset()[0]

# Keep track of the total reward
total_reward = np.zeros(env.reward_space.shape[0])

# Initialize done and absorbing
done = False
absorbing = False

for i in range(5):  # Run for 5 tasks
    while not (done or absorbing):
        # Get the current state index
        state_index = get_state_index(state)

        # Choose action with highest average reward
        action = np.argmax([np.mean([np.sum(solution) for solution in Q_set[state_index][a]]) if len(Q_set[state_index][a]) > 0 else 0 for a in range(n_actions)])

        # Take the action and get the reward
        next_state, reward, done, absorbing, _ = env.step(action)

        # Add the reward to the total reward
        total_reward += reward

        # Move to the next state
        state = next_state

    # Print total reward
        print("Total reward for task ", i + 1, ": ", total_reward)

    # Reset the environment for the next task
    state = env.reset()[0]
    done = False
    total_reward = np.zeros(env.reward_space.shape[0])


