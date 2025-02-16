#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# In[2]:


env_name = 'Breakout-v0'
env = gym.make(env_name)


# In[3]:


frames = [] # array to store state space at each step

env.reset()
done = False
for _ in range(300): 
    #print(done)
    frames.append(env.render(mode='rgb_array'))
    obs,reward,done, _ = env.step(env.action_space.sample())
    if done:
        break


# In[4]:


patch = plt.imshow(frames[0])
plt.axis('off')
def animate(i):
    patch.set_data(frames[i])
anim = animation.FuncAnimation(plt.gcf(), animate, \
        frames=len(frames), interval=10)
anim.save('random_agent.gif', writer='imagemagick')


# In[5]:


import gym
env = gym.make("Breakout-v0")
env = gym.wrappers.Monitor(env, 'recording', force=True)
observation = env.reset()
for _ in range(1000):
    #env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()


# In[ ]:




