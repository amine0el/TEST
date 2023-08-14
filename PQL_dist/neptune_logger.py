import os
import shutil
import tempfile

import torch
from time import time
from matplotlib import pyplot as plt
import neptune as neptune

from utils import save_to_yml, load_yml
from configs.constants import *
from configs.strings import *


class NeptuneLogger:
    """

    """

    def __init__(self):
        self.run = None
        self.init_time = int(time())

    def log_params(self, params):
        """

        Parameters
        ----------
        params
        """
        self.run[PARAMETERS] = params

    def log_config(self, config):
        """

        Parameters
        ----------
        config: dict
        """
        self.run[CONFIG] = config

    def log_metric(self, metric_value, metric_name, mode, step=None):
        """

        Parameters
        ----------
        metric_value
        metric_name
        mode
        step
        """
        self.run[f'{mode}/{metric_name}'].log(metric_value, step=step)

    def log_fig(self, fig, fig_name, mode, step=None):
        """

        Parameters
        ----------
        fig
        fig_name
        mode
        step
        """
        self.run[f'{mode}/{fig_name}'].log(fig, step=step)
        plt.clf()
        plt.close("all")

    def upload_config(self, config, model_version=None):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = save_to_yml(config, CONFIG_FILE, tmp_dir)
            if model_version is None:
                self.run[f"{MODEL}/{CONFIG}"].upload(config_path, wait=True)
            else:
                model_version[f"{MODEL}/{CONFIG}"].upload(config_path, wait=True)

    def upload_fig(self, fig, fig_name, mode):
        """

        Parameters
        ----------
        fig
        fig_name
        mode
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = os.path.join(tmp_dir, f'{fig_name}.png')
            plt.savefig(img_path)
            self.run[f'{mode}/{fig_name}_ul'].upload(img_path, wait=True)
            plt.clf()
            plt.close("all")

    def _log_model(self, model_name, model):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, f"{model_name}.mod")
            torch.save(model, model_path)
            self.run[f"{MODEL}/{model_name}"].upload(model_path, wait=True)

    def _log_model_metrics(self, model_version, model, config):
        """

        Parameters
        ----------
        model_version
        model
        config
        """
        model_version["run/id"] = self.run["sys/id"].fetch()
        model_version["run/url"] = self.run.get_url()
        # model_version["run/val_class_acc"] = self.run["metric/aggr/val_class_acc"].fetch_last()
        # model_version["run/val_sample_acc"] = self.run["metric/aggr/val_sample_acc"].fetch_last()
        # model_version[f"{MODEL}/classes"] = model.classes
        # model_version[f"{MODEL}/img_size"] = model.img_size
        # model_version[f"{MODEL}/n_num_features"] = model.n_num_features
        if config is not None:
            self.upload_config(config, model_version)
            # model_version[f"{MODEL}/config"] = config

    def log_models(self, models, config):
        """

        Parameters
        ----------
        models
        """
        self.upload_config(config)
        for model_name, model in models.items():
            self._log_model(model_name, model['model'])

    def log_dataframe(self, df, df_name, mode):
        """

        Parameters
        ----------
        df
        df_name
        mode
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, f'{df_name}.csv')
            df.to_csv(csv_path)
            self.run[f'{mode}/{df_name}'].upload(csv_path, wait=True)

    def fetch_values(self, folder, name):
        return self.run[f"{folder}/{name.split('.')[0]}"].fetch_values()

    def download_metrics(self, folder, name, save_path):
        file_path = os.path.join(save_path, f'{name}.csv')
        df = self.fetch_values(folder, name)
        df.to_csv(file_path)
        return file_path

    def download_artifact(self, folder, name, save_path):
        """

        Parameters
        ----------
        model_config_dict
        base_path
        tmp_path

        Returns
        -------

        """
        file_path = os.path.join(save_path, name)
        self.run[f"{folder}/{name.split('.')[0]}"].download(save_path)
        return file_path

    def download_model(self, model_type):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = self.download_artifact('model', f'{model_type}.mod', tmp_dir)
            config_path = self.download_artifact('model', f'{CONFIG}.yml', tmp_dir)
            model = torch.load(model_path)
            config = load_yml(config_path)
        return model, config

    def start(self, run_name, tags, run_id=None):
        """

        Parameters
        ----------
        run_name
        tags
        """
        if run_id is not None:
            self.run = neptune.init_run(project=PROJECT, with_id=run_id)
        else:
            self.run = neptune.init_run(project=PROJECT, tags=tags, name=run_name)

    def stop(self):
        """

        """
        self.run.stop()
