import time
import numpy as np
import gym
import gridworld
from gridworld import CliffWalkingWapper

class SarsaAgent():

 def __init__(self,n_states,n_act,e_greed=0.1,lr=0.1,gamma=0.9):
    self.e_greed=e_greed
    self.Q=np.zeros((n_states,n_act))
    self.n_act=n_act
    self.lr=lr
    self.gamma=gamma


 def predict(self,state):
     Q_list=self.Q[state,:]
     action=np.random.choice(np.flatnonzero(Q_list==Q_list.max()))#取最大值
     return action


#forward
 def act(self,state):
    if np.random.uniform(0,1)<self.e_grees:#探索
     action=np.random.choice(self.n_act)
    else:# 利用
      action=self.predict(state)
    return action
   
    

#backward
 def learn(self,state,action,reward,next_state,next_action,done):
   cur_Q=self.Q[state,action]

   if done:
      target_Q=reward
   else: 
       target_Q=self.gamma*self.Q[next_state,next_action]
   self.Q[state,action]+=self.lr*(target_Q-cur_Q)


def train_episode(env,agent,is_render):
   state=env.reset()
   action=agent.act(state)
   total_reward=0
   while True:
      next_state,reward,done,_=env.step(action)
      next_action=agent.act(next_state)

      agent.learn(state,action,reward,next_state,next_action,done)

      action=next_action
      state=next_state
      total_reward+=reward


      if is_render:env.render()#图形窗口可视化
      if done:break

      return total_reward



def test_episode(env,agent):
   state=env.reset()
 
   total_reward=0
   while True:
      action=agent.predict(state)
      next_state,reward,done,_=env.step(action)
      



      
      state=next_state
      total_reward+=reward
      env.render()
      time.sleep((0.5))

      if done:break

      return total_reward
   

def train(env,episodes=500,lr=0.1,gamma=0.9,e_greed=0.1):
    agent= SarsaAgent(
       n_states=env.observation_space.n,
       n_act=env.action_space.n,
       lr=lr,
       gamma=gamma,
       e_greed=e_greed)
    
    is_render=False
    for e in range(episodes):
       ep_reward=train_episode(env,agent,is_render)
       print('Episode %s:reward=%.1f'%(e,ep_reward))

       if e%50==0:
          is_render=True
       else:
          is_render=False

    test_reward=test_episode(env,agent)
    print('test_reward=%.1f'%(test_reward))

if __name__=='__main__':
   env=gym.make("CliffWalking-v0")
   env=gridworld.CliffWalkingWapper(env)
   train(env)
