import gym
import numpy as np
import tensorflow as tf

from nets.agent import Agent


def Testing(num_frames=4):
    """Testing mode

    Testing mode is using for evaluate agent's score in the testing game environment. 
    The agent isn't training during this process. 
    Since testing is running, the statistics about agent's performace is reporting.

    Arguments:
        num_frames (int): Number of frames in state shape
    """
    env = gym.make('CarRacing-v0')
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    print(f"State: {state_dim}")
    print(f"Action: {action_dim}")

    # load Agent's model
    a = Agent()
    a.load('save/model.h5')

    # logger
    ep_ret_log = []

    # init environment
    obs, ep_ret, ep_len, epoch = env.reset(), 0, 0, 0
    obs = np.expand_dims(obs, axis=0)
    state_stack = np.repeat(obs, num_frames, axis=0)
    print(state_stack.shape)
    print(state_stack.dtype)

    # main loop
    while True:
        # render window
        env.render()

        # take action
        act = a.act(state_stack)
        act = tf.concat([act[0], act[1]], axis=-1)
        act = tf.squeeze(act).numpy()
        print(act)

        obs2, r, d, _ = env.step(act)
        obs2 = np.expand_dims(obs2, axis=0)
        state_stack = np.append(state_stack[1:], obs2, axis=0)

        # statistics
        ep_ret += r
        ep_len += 1

        # End of episode
        if d:
            print(f"Epoch: {epoch}, EpRet: {ep_ret}, EpLen: {ep_len}")

            # if exists statistical data
            if len(ep_ret_log) > 0:
                log = np.array(ep_ret_log)
                print("AvgEpRet:", log.mean())
                print("StdEpRet:", log.std())
                print("MaxEpRet:", log.max())
                print("MinEpRet:", log.min())

            ep_ret_log.append(ep_ret)

            obs, ep_ret, ep_len = env.reset(), 0, 0
            obs = np.expand_dims(obs, axis=0)
            state_stack = np.repeat(obs, num_frames, axis=0)

            epoch += 1
