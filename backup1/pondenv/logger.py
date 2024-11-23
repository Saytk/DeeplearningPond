from torch.utils.tensorboard import SummaryWriter
import os


class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def log_episode(self, episode, reward, epsilon):
        self.writer.add_scalar("Reward/Episode", reward, episode)
        self.writer.add_scalar("Exploration/Epsilon", epsilon, episode)

    def close(self):
        self.writer.close()
