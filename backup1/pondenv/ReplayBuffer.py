import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the Replay Buffer.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state.
            done (bool): Whether the episode ended.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Batch of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns:
            int: Number of transitions stored in the buffer.
        """
        return len(self.buffer)
