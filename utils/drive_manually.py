import gym
import numpy as np

from pyglet.window import key
from utils.memory import Memory

def key_press(k, mod):
    global exit_game
    global actions
    
    if k == key.ESCAPE: exit_game = True
    elif k == key.UP:    
        actions[3] = +1.0
        if actions[0] == 0.0 and actions[2] == 0.0:
            actions[1] = +1.0
    elif k == key.LEFT:  
        actions[0] = -1.0
        actions[1] =  0.0  # Cut gas while turning
    elif k == key.RIGHT: 
        actions[0] = +1.0
        actions[1] =  0.0  # Cut gas while turning
    elif k == key.DOWN:  
        actions[2] = +0.4  # stronger brakes
        actions[1] =  0.0  # Cut gas while braking

def key_release(k, mod):
    if k == key.LEFT and actions[0] == -1.0: 
        actions[0] = 0.0
        if actions[3] == 1.0:
            actions[1] = 1.0
    elif k == key.RIGHT and actions[0] == +1.0: 
        actions[0] = 0.0
        if actions[3] == 1.0:
            actions[1] = 1.0
    elif k == key.UP:    
        actions[1] = 0.0
        actions[3] = 0.0
    elif k == key.DOWN:  
        actions[2] = 0.0

# exit game on keyboard
exit_game = False

def Game(max_ep_len=1000, num_frames=4):
    global exit_game
    global actions

    env = gym.make('CarRacing-v0')
    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    print(f"State: {state_dim}")
    print(f"Action: {action_dim}")

    # set interrupts
    env.reset()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    # make global actions array 
    actions = np.zeros(4, dtype=np.float32)

    # mem
    memory = Memory()
    memory.create(state_dim, action_dim)

    # logger
    ep_ret_log = []

    # init environment
    obs, ep_ret, ep_len, epoch = env.reset(), 0, 0, 0
    obs = np.expand_dims(obs, axis=0)
    state_stack = np.repeat(obs, num_frames, axis=0)
    print(state_stack.shape)
    print(state_stack.dtype)

    # main loop
    while exit_game == False:
        # render window
        env.render()

        # take action
        obs2, r, d, _ = env.step(actions[:3])
        obs2 = np.expand_dims(obs2, axis=0)
        state_stack = np.append(state_stack[1:], obs2, axis=0)

        # statistics
        ep_ret += r
        ep_len += 1

        # Ignore the 'done' signal
        d = False if ep_len == max_ep_len else d

        # store in memory
        memory.add(state_stack, np.array(actions[:3]), r, d)
        
        # End of episode
        if d or (ep_len == max_ep_len):
            print(f"Epoch: {epoch}, EpRet: {ep_ret}, EpLen: {ep_len}, ReplayBuff: {len(memory)}")

            # if exists statistical data
            if len(ep_ret_log) > 0:
                log = np.array(ep_ret_log)
                print("AvgEpRet:", log.mean())
                print("StdEpRet:", log.std())
                print("MaxEpRet:", log.max())
                print("MinEpRet:", log.min())
            
            print()

            ep_ret_log.append(ep_ret)

            obs, ep_ret, ep_len = env.reset(), 0, 0
            obs = np.expand_dims(obs, axis=0)
            state_stack = np.repeat(obs, num_frames, axis=0)

            epoch += 1
    
    print('\n')

    # save the dataset
    memory.save()