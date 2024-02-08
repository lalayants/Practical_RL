import multiprocessing
import gymnasium as gym
import numpy as np
from threading import Thread

class CustomThread(Thread):
    # constructor
    def __init__(self):
        # execute the base constructor
        Thread.__init__(self)
    # function executed in a new thread
    def run(self, env, agent, i, dict):
        print(i, 'start')
        dict[i] = generate_session(env, agent)
        print(i, 'done')


def generate_session(env, agent, t_max=1000):
    """
    Play a single game using agent neural network.
    Terminate when game finishes or after :t_max: steps
    """
    states, actions = [], []
    total_reward = 0
    
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    s, _ = env.reset()

    for t in range(t_max):
        probs = agent.predict_proba(s.reshape(1, -1) ).reshape(-1) 
        assert probs.shape == (env.action_space.n,), "make sure probabilities are a vector (hint: np.reshape)"
        a = np.random.choice(np.arange(n_actions), p=probs)

        new_s, r, terminated, truncated, _ = env.step(a)

        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if terminated or truncated:
            break
    return states, actions, total_reward


def generate_session_dict(env, agent, i, return_dict=''):
    # env = gym.make(env, render_mode="rgb_array").env
    # print(i, 'start')
    return generate_session(env, agent)
    # print(i, 'done')


def generate_sessions(env, agent, n):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(n):
        p = multiprocessing.Process(target=generate_session_dict, args=(env, agent, i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    # print(return_dict.values())
    return return_dict.values()