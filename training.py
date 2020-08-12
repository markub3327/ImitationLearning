import os
import numpy as np
import wandb

from nets.agent import Agent
from wandb.keras import WandbCallback

def Training(datasets='data/', hid=[32, 64], num_frames=4):
    
    def load_datasets():
        # the list of datasets
        dat = []
        
        # scan datasets
        with os.scandir(datasets) as entries:
            for entry in entries:
                dat.append(entry.path)
        
        return dat

    def read_dataset(path):
        with np.load(path) as data:
            f = data['frames']
            a = data['actions']
            print(f.shape)
            print(a.shape)

            # make timesteps
            f, a = make_timesteps(f, a, timesteps=num_frames)
            print(f.shape)
            print(a.shape)

            # shuffle dataset after loading from file
            f, a = shuffle_dataset(f, a)
        
        return f, a

    def make_timesteps(f_dat, a_dat, timesteps):
        # generate random indexes
        rand_idxs = np.arange(timesteps + 1, f_dat.shape[0], dtype=np.int)
        print(rand_idxs)
        print(rand_idxs.shape)

        states = np.zeros((rand_idxs.shape[0], timesteps) + f_dat.shape[1:], dtype=np.uint8)
        #next_states = np.zeros((BATCH_SIZE, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES), dtype=np.float32)
            
        for i, idx in enumerate(rand_idxs):
            states[i] = f_dat[idx-timesteps-1:idx-1]
            #next_states[i] = self._frames[idx-num_frames:idx]

        return states, a_dat[rand_idxs] #, self._rewards[rand_idxs], next_states, self._terminal[rand_idxs]
        
    def shuffle_dataset(f_dat, a_dat):
        idx = np.arange(0, f_dat.shape[0], dtype=np.int)
        print(idx)
        np.random.shuffle(idx)
        print(idx)

        return f_dat[idx], a_dat[idx]

    wandb.init(project="car_racing")

    # create network
    agent = Agent()
    agent.create((num_frames, 96, 96, 3), hid=hid)

    # save model's plot
    agent.save_plot()
    
    # load datasets from folder
    datasets = load_datasets()

    # take every dataset from folder
    for dat in datasets:
        print('+-----------------------------------------------+')

        print(f"Loading data from dataset: {dat}")
        f, a = read_dataset(dat)

        print('Run training...')
        print('|-----------------------------------------------|')
        agent.train(f, a, epochs=1000, top_only=True, callbacks=[WandbCallback()])
        
        # re-shuffle dataset before fine-tuning
        f, a = shuffle_dataset(f, a)

        print('Run fine-tuning...')
        print('|-----------------------------------------------|')
        agent.train(f, a, epochs=1000, top_only=False, callbacks=[WandbCallback()])

        print('+-----------------------------------------------+')

    # save model
    agent.save()

