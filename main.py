from ddpg import DDPG
from env_zeromap_wukong import Env
import time
import numpy as np

env = Env()
agent = DDPG(6, 7)

t1 = time.time()
replay_num = 0
success = np.zeros(10000)
for i in range(10000):
    t_start = time.time()
    sd = i * 3 + 100
    s = env.set_state_seed(sd)
    agent.exploration_noise.reset()
    ep_reward = 0
    ave_w = 0
    j = 0
    r = 0
    for j in range(200):
        # Add exploration noise
        a = agent.noise_action(s)
        a_store = a.copy()
        ave_w += np.linalg.norm(a[-3:])
        a[:4] /= max(np.linalg.norm(a[:4]), 1e-8)
        a[-3:] *= 2
        a = np.minimum(2, np.maximum(-2, a))

        s_, r, done = env.step(a)
        agent.perceive(s, a_store, r, s_, done)
        replay_num += 1
        s = s_
        ep_reward += r

        if done:
            if r == 10:
                success[i] = 1
            break
    ave_w /= j+1
    print("episode: %6d   ep_reward:%8.5f   last_reward:%6.5f   replay_num:%8d   cost_time:%4.2f    ave_w:%8.5f    "
          "success_rate:%4f" % (i, ep_reward, r, replay_num, time.time() - t_start, ave_w, sum(success)/(i+1)))
print('Running time: ', time.time() - t1)
