from matplotlib import pyplot as plt
import seaborn as sns
# NEEDED!! for 3d plotting
from mpl_toolkits.mplot3d import Axes3D


class Visualization(object):
    """

    """

    def __init__(self, n_obj, front=None):
        self.front = front
        self.n_obj = n_obj
        if n_obj == 2:
            self.gen_xy_plot = self.gen_xy_plot_2d
        else:
            self.gen_xy_plot = self.gen_xy_plot_nd

    def set_front(self, front):
        """

        Parameters
        ----------
        front
        """
        self.front = front

    def _gen_plt_2obj(self, fig, u, au, u_itr, au_itr, y, yf, itr, emp_front):
        ax = fig.add_subplot(111)
        picsize = fig.get_size_inches() / 1.3
        fig.set_size_inches(picsize)
        ax.cla()
        if self.front is not None:
            ax.plot(self.front[:, 0], self.front[:, 1], marker='o', linestyle='None', color='b', markersize=8,
                    label=r'$Frontier_{True}$')
        if emp_front is not None:
            ax.plot(emp_front[:, 0], emp_front[:, 1], marker='*', linestyle='None', color='black',
                    markersize=8,
                    label=r'$Frontier_{Empiric}$')
        ax.plot(y[:, 0], y[:, 1], marker='o', linestyle='None', color='r', markersize=2, label=r'$Samples$')
        ax.plot(yf[:, 0], yf[:, 1], marker='o', linestyle='None', color='g', markersize=4,
                label=r'$Frontier_{Approx}$')
        if u is not None:
            ax.plot(u[0], u[1], marker='s', linestyle='None', color='orange', markersize=8)
        if au is not None:
            ax.plot(au[0], au[1], marker='s', linestyle='None', color='orange', markersize=8)
        if u_itr is not None:
            ax.plot(u_itr[0], u_itr[1], marker='x', linestyle='None', color='orange', markersize=8)
        if au_itr is not None:
            ax.plot(au_itr[0], au_itr[1], marker='x', linestyle='None', color='orange', markersize=8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.title(str(itr))
        return fig

    def _gen_plt_3obj(self, fig, u, au, u_itr, au_itr, y, yf, itr, emp_front):
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')
        picsize = fig.get_size_inches() / 1.3
        picsize[0] *= 2
        fig.set_size_inches(picsize)
        ax1.cla()
        if self.front is not None:
            ax1.plot(self.front[:, 0], self.front[:, 1], self.front[:, 2], marker='o', linestyle='None',
                     color='b', markersize=8, label=r'$Frontier_{True}$')
        if emp_front is not None:
            ax1.plot(emp_front[:, 0], emp_front[:, 1], emp_front[:, 2], marker='*', linestyle='None',
                     color='black', markersize=8, label=r'$Frontier_{Empiric}$')
        ax1.plot(y[:, 0], y[:, 1], y[:, 2], marker='o', linestyle='None', color='r', markersize=2,
                 label=r'$Samples$')
        ax1.plot(yf[:, 0], yf[:, 1], yf[:, 2], marker='o', linestyle='None', color='g', markersize=4,
                 label=r'$Frontier_{Approx}$')
        if u is not None:
            ax1.plot(u[0], u[1], u[2], marker='s', linestyle='None', color='orange', markersize=8)
        if au is not None:
            ax1.plot(au[0], au[1], au[2], marker='s', linestyle='None', color='orange', markersize=8)
        if u_itr is not None:
            ax1.plot(u_itr[0], u_itr[1], u_itr[2], marker='s', linestyle='None', color='orange', markersize=8)
        if au_itr is not None:
            ax1.plot(au_itr[0], au_itr[1], au_itr[2], marker='x', linestyle='None', color='orange',
                     markersize=8)
        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        ax2.cla()
        ax2.plot(y[:, 0], y[:, 1], y[:, 2], marker='o', linestyle='None', color='r', markersize=4)
        ax2.plot(yf[:, 0], yf[:, 1], yf[:, 2], marker='o', linestyle='None', color='g', markersize=2)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.view_init(elev=50., azim=25)  # change view for second plot
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        plt.suptitle(str(itr))
        return fig

    def gen_plt(self, u, au, u_itr, au_itr, y, yf, itr, emp_front=None):
        """

        Parameters
        ----------
        u
        au
        u_itr
        au_itr
        y
        yf
        itr
        emp_front

        Returns
        -------

        """
        fig = plt.figure()
        if self.n_obj == 2:
            fig = self._gen_plt_2obj(fig, u, au, u_itr, au_itr, y, yf, itr, emp_front)
        elif self.n_obj == 3:
            fig = self._gen_plt_3obj(fig, u, au, u_itr, au_itr, y, yf, itr, emp_front)
        plt.legend()
        plt.grid(True)
        return fig

    def gen_hist(self, params, itr, name):
        """

        Parameters
        ----------
        params
        itr
        name

        Returns
        -------

        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.histplot(params, ax=ax)
        plt.title(f'{name} - {itr}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        return fig

    def gen_xy_plot_nd(self, x, y, itr, name, x_front=None, x_front_emp=None):
        """

        Parameters
        ----------
        x
        y
        itr
        name
        x_front
        x_front_emp

        Returns
        -------

        """
        x_dim = 0
        fig = plt.figure()
        # plt.plot(x[:, 0], y.sum(axis=1), marker='*', linestyle='None', markersize=8, color='g', label=r'$\sum Objective$')
        for i in range(self.n_obj):
            plt.plot(x[:, x_dim], y[:, i], marker='o', linestyle='None', markersize=4,
                     label=r'$Objective_{}$'.format(i))
            if i == 0:
                if x_front is not None:
                    plt.plot(x[:, x_dim], x_front[:, i], color='black', label=r'$Front_{True}$')
                if x_front_emp is not None:
                    plt.plot(x[:, x_dim], x_front_emp[:, i], color='gray', label=r'$Front_{Empirical}$',
                             linestyle='-')
            else:
                if x_front is not None:
                    plt.plot(x[:, x_dim], x_front[:, i], color='black')
                if x_front_emp is not None:
                    plt.plot(x[:, x_dim], x_front_emp[:, i], color='gray')
        plt.title(f'{name} - {itr}')
        plt.ylabel('Reward')
        plt.xlabel('Preference for Obj1')
        plt.legend()
        return fig

    def gen_xy_plot_2d(self, x, y, itr, name, x_front=None, x_front_emp=None):
        """

        Parameters
        ----------
        x
        y
        itr
        name
        x_front
        x_front_emp

        Returns
        -------

        """
        x_dim = 0
        fig, ax1 = plt.subplots()
        # plt.plot(x[:, 0], y.sum(axis=1), marker='*', linestyle='None', markersize=8, color='g', label=r'$\sum Objective$')
        i = 0
        color = 'tab:orange'
        ax1.plot(x[:, x_dim], y[:, i], marker='o', linestyle='None', markersize=4,
                 #label=r'$Objective_{}$'.format(i),
                 color=color)
        ax1.set_ylabel(f'Reward Obj{i}', color=color)
        # ax1.set_ylabel(r'$Reward Obj_{}$'.format(i), color=color)
        if x_front is not None:
            ax1.plot(x[:, x_dim], x_front[:, i], color=color, linestyle='--')
        if x_front_emp is not None:
            ax1.plot(x[:, x_dim], x_front_emp[:, i], color='gray',linestyle='-')
        ax1.set_xlabel('Preference for Obj0')

        i = 1
        color = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.plot(x[:, x_dim], y[:, i], marker='o', linestyle='None', markersize=4,
                 #label=r'$Objective_{}$'.format(i),
                 color=color)

        ax2.set_ylabel(f'Reward Obj{i}', color=color)
        # ax2.set_ylabel(r'$Reward Obj_{}$'.format(i), color=color)
        if x_front is not None:
            ax2.plot(x[:, x_dim], x_front[:, i], color='black', linestyle='--', label='Optimal Rewards')
            ax2.plot(x[:, x_dim], x_front[:, i], color=color, linestyle='--')
        if x_front_emp is not None:
            ax2.plot(x[:, x_dim], x_front_emp[:, i], color='gray')
        plt.title(f'{name} - {itr}')
        plt.legend(loc='lower center')
        plt.tight_layout()
        return fig


if __name__ == '__main__':
    import pandas as pd

    data_path = '/Users/eikementzendorff/Downloads/results_10000.csv'
    data = pd.read_csv(data_path, index_col=0)
    ef_data_path = '/Users/eikementzendorff/Downloads/empiric_frontier_10000.csv'
    ef_data = pd.read_csv(ef_data_path, index_col=0)
    vis = Visualization(2, ef_data.values)
    vis.gen_xy_plot_2d(data[['pref_0', 'pref_1']].values, data[['reward_0', 'reward_1']].values, 25500, 'test', x_front_emp=ef_data.values)
    plt.show()
    pass
