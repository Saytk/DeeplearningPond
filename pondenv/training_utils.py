import os
import torch
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    """
    A utility class to log training metrics to TensorBoard and save results.
    """
    def __init__(self, log_dir="runs"):
        self.writer = SummaryWriter(log_dir)

    def log_metric(self, metric_name, value, step):
        """
        Logs a scalar metric to TensorBoard.

        Args:
            metric_name (str): Name of the metric.
            value (float): Value of the metric.
            step (int): Training step or episode.
        """
        self.writer.add_scalar(metric_name, value, step)

    def close(self):
        """
        Closes the TensorBoard writer.
        """
        self.writer.close()

    def save_model(self, model, model_path):
        """
        Saves a PyTorch model.

        Args:
            model (torch.nn.Module): Model to save.
            model_path (str): Path to save the model.
        """
        torch.save(model.state_dict(), model_path)

    def load_model(self, model, model_path):
        """
        Loads a PyTorch model.

        Args:
            model (torch.nn.Module): Model to load weights into.
            model_path (str): Path from which to load the model.
        """
        model.load_state_dict(torch.load(model_path))
        model.eval()
