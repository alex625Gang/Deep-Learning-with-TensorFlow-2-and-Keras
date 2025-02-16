#!/usr/bin/env python
# coding: utf-8

# In[1]:


from gym import envs
all_envs = envs.registry.all()
env_ids = [env_spec.id for env_spec in all_envs]
print(*env_ids, sep='\n')


# In[2]:


import gym


# In[3]:


env_name = 'Breakout-v0'
env = gym.make(env_name)


# In[4]:


obs = env.reset()
env.render()


# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
obs = env.reset()
plt.imshow(env.render(mode='rgb_array'))


# In[6]:


print(env.observation_space)


# In[7]:


print(env.action_space)


# In[8]:


env.close()


# In[ ]:




