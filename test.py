import array
import numpy as np
import pandas as pd


def get_eval_w(n, dim):
    """

    Parameters
    ----------
    n
    dim

    Returns
    -------

    """
    step = 1 / n
    b = []
    if dim == 4:
        for b1 in np.arange(0, 1 + .00000001, step):
            b234 = 1 - b1
            for b2 in np.arange(0, b234 + .00000001, step):
                b34 = b234 - b2
                for b3 in np.arange(0, b34 + .00000001, step):
                    b4 = b34 - b3
                    b += [[b1, b2, b3, abs(b4)]]
    elif dim == 5:
        for b0 in np.arange(0, 1 + .00000001, step):
            b2345 = 1 - b0
            for b1 in np.arange(0, b2345 + .00000001, step):
                b234 = b2345 - b1
                for b2 in np.arange(0, b234 + .00000001, step):
                    b34 = b234 - b2
                    for b3 in np.arange(0, b34 + .00000001, step):
                        b4 = b34 - b3
                        b += [[b0, b1, b2, b3, abs(b4)]]
    elif dim == 6:
        for b1m in np.arange(0, 1 + .00000001, step):
            b23456 = 1 - b1m
            for b0 in np.arange(0, b23456 + .00000001, step):
                b2345 = b23456 - b0
                for b1 in np.arange(0, b2345 + .00000001, step):
                    b234 = b2345 - b1
                    for b2 in np.arange(0, b234 + .00000001, step):
                        b34 = b234 - b2
                        for b3 in np.arange(0, b34 + .00000001, step):
                            b4 = b34 - b3
                            b += [[b1m, b0, b1, b2, b3, abs(b4)]]
    elif dim == 3:
        for b1 in np.arange(0, 1 + .00000001, step):
            b234 = 1 - b1
            for b2 in np.arange(0, b234 + .00000001, step):
                b3 = b234 - b2
                b += [[b1, b2, abs(b3)]]
    elif dim == 2:
        for b1 in np.arange(0, 1 + .00000001, step):
            b2 = 1 - b1
            b += [[b1, abs(b2)]]
    elif dim == 1:
        b = [[1]]
    return np.array(b)

def calc_opt_reward(prefs, front, u_func=None):
    """

    Parameters
    ----------
    prefs
    front

    Returns
    -------

    """
    if u_func is None:
        u_func = lambda x, y: (x * y).sum(axis=1)
    prefs = np.float32(prefs)
    w_front = []
    for n, w in enumerate(prefs):
        id = u_func(front, w).argmax()
        w_front.append([w,front[id]])
    return w_front

# def get_wfront(self,n=32):
#         eval_w = get_eval_w(n, 2)
#         rewards = np.array([[0.7, -1],[8.2,-3],[11.5,-5],[14,-7],[15.1,-8],[16.1,-9],[19.6,-13],[20.3,-14],[22.4,-17],[23.7,-19]])
    
#         rewards_1 = rewards[:,0]
#         rewards_2 = []
#         for i in range(1, int(((rewards[:,1])*-1).max()+1)):
#             rewards_2.append(i)
#         rewards_2=np.array(rewards_2)
#         t_func = lambda x, y: (x / y)

#         u_func = lambda x, y: abs((x - y))
#         nf1 = t_func(rewards_1, (rewards_1.max(axis=0)))
#         nf2 = t_func(rewards_2, (rewards_2.max(axis=0)))
#         nf1 = np.float32(nf1)
#         nf2 = np.float32(nf2)
#         returns = []
#         horizons = []
#         for n , w in enumerate(eval_w):
#             id_1 = u_func(nf1 , w[0]).argmin()
#             id_2 = u_func(nf2 , w[1]).argmin()
#             returns.append([rewards_1[id_1], rewards_2[id_2]-20])
#             horizons.append(abs(rewards_2[id_2]-20))
#         return returns, horizons



def get_wfront(n=32):
        prefs = get_eval_w(n, 2)
        front = np.array([[0.7, -1],[8.2,-3],[11.5,-5],[14,-7],[15.1,-8],[16.1,-9],[19.6,-13],[20.3,-14],[22.4,-17],[23.7,-19]])
    
        rewards_1 = front[:,0]
        rewards_2 = front[:,1]*-1
    
        t_func = lambda x, y: (x / y)

        u_func = lambda x, y: abs((x - y))
        nf1 = t_func(rewards_1, (rewards_1.max(axis=0)))
        nf2 = t_func(rewards_2, (rewards_2.max(axis=0)))
        nf1 = np.float32(nf1)
        nf2 = np.float32(nf2)
        returns = []
        horizons = []
        for n , w in enumerate(prefs):
            id_1 = u_func(nf1 , w[0]).argmin()
            id_2 = u_func(nf2 , w[1]).argmin()
            returns.append([rewards_1[id_1], rewards_2[id_2]-20])
            horizons.append(abs(rewards_2[id_2]-20))
        return eval_w, returns, horizons


if __name__ == '__main__':
    #eval_w = get_eval_w(32, 2)
    rewards = np.array([[0.7, -1],[8.2,-3],[11.5,-5],[14,-7],[15.1,-8],[16.1,-9],[19.6,-13],[20.3,-14],[22.4,-17],[23.7,-19]])
    evalw, returns, horizons = get_wfront(32)
    df = [[evalw[x],horizons[x],returns[x]] for x in range(len(returns))]
    df = pd.DataFrame()
    df.to_csv('mmm_front.csv')

    # rewards_1 = []
    # for n in rewards:
    #     rewards_1.append([n[0]])
    # rewards_2 = []   
    # for n in range(1,20):
    #     rewards_2.append([n])
    # rewards_1 = np.array(rewards_1)
    # rewards_2 = np.array(rewards_2)
    
    # t_func = lambda x, y: (x / y)
    # u_func = lambda x, y: abs((x - y))
    # nf1 = t_func(rewards_1, rewards_1.max(axis=0))
    # nf2 = t_func(rewards_2, rewards_2.max(axis=0))
    # nf1 = np.float32(nf1)
    # nf2 = np.float32(nf2)
    # returns = []
    # horizon = []
    # for n , w in enumerate(eval_w):
    #     id_1 = u_func(nf1 , w[0]).argmin()
    #     id_2 = u_func(nf2 , w[1]).argmin()
    #     returns.append([rewards_1[id_1], rewards_2[id_2]-20])
    #     horizon.append(rewards_2[id_2])
    

    # df = pd.DataFrame(w_front)
    # print(df.head())
    # df.to_csv('test2.csv')