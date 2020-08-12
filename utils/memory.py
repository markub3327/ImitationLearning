import numpy as np

class Memory:
    """Replay buffer memory for training agent from samples
    """
    def __init__(self):
        self._len = 0

    def create(self, state_dim, action_dim, limit=1000000):
        self._limit = limit

        # create arrays for storing samples
        self._frames = np.zeros((limit,) + state_dim, dtype=np.uint8)
        self._actions = np.zeros((limit,) + action_dim, dtype=np.float32)
        self._rewards = np.zeros(limit, dtype=np.float32)
        self._terminals = np.zeros(limit, dtype=np.bool)

    def add(self, obs, action, reward, terminal):
        self._frames[self._len] = obs[0]
        self._actions[self._len] = action
        self._rewards[self._len] = reward
        self._terminals[self._len] = terminal
        #if self._i % (self._max_memory - 1) == 0 and self._i != 0:
        #    self._i = BATCH_SIZE + NUM_FRAMES + 1
        #else:
        self._len += 1

    def save(self, fileName='data.npz'):
        print(self._len)

        # uloz matice numpy v rozsahu 0 az dlzka buffer-u
        np.savez_compressed(fileName, frames=self._frames[:self._len], actions=self._actions[:self._len], rewards=self._rewards[:self._len], terminals=self._terminals[:self._len])
        print(f"Saved successfully '{fileName}'")

    def __len__(self):
        return self._len