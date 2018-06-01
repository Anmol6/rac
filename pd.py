import numpy as np
from env import mControl
import ipdb

ob_pos = 10.0
goal_pos = 20.0
ob_vel = 0.0
mass = 10.0
discount=0.9
render = False
#env = mControl(goal_pos=goal_pos, ob_pos=ob_pos, ob_vel=ob_vel, render=render)

ctrl_loops = 1000

def ctrl_fn(x, v, kp=60, kd=35):
    return -kp*x -kd*v

env = mControl(goal_pos=goal_pos, ob_pos=ob_pos, ob_vel=ob_vel, mass=mass, render=render)

'''
for i in range(ctrl_loops):
    x_ctrl = ob_pos-goal_pos
    v_ctrl = ob_vel 
    state,r,done,_ = env.step(ctrl_fn(x_ctrl,v_ctrl))
    ob_pos = state[0]
    ob_vel = state[1]
    if done:
        break
    #reward_total+=(discount**i)*r


'''







kd_list = np.linspace(1,2,49)
kp_list = np.linspace(1,2,49)
#kd_list = [35]
#kp_list=[60]
def ctrl(kp, kd, ob_pos=ob_pos, goal_pos=goal_pos, mass=mass, ob_vel=ob_vel, render=render):
    reward_total=0
    env = mControl(goal_pos=goal_pos, ob_pos=ob_pos, ob_vel=ob_vel, mass=mass, render=render)
    for i in range(ctrl_loops):
        #print(i)
        x_ctrl = ob_pos-goal_pos
        v_ctrl = ob_vel 
        state,r,done,_ = env.step(ctrl_fn(x_ctrl,v_ctrl, kp=kp, kd=kd))
        ob_pos = state[0,0]
        ob_vel = state[0,1]
        if done:
            break
        reward_total+=(discount**i)*r
    #del env
    return reward_total



kp_kd_v_list=[]

for kd in kd_list:
    for kp in kp_list:
        print('kp: ' + str(kp) +', kd: ' + str(kd))
        V = ctrl(kd,kp)
        #ipdb.set_trace()
        print('V: ' + str(V))
        kp_kd_v_list.append(np.array([kp,kd,V]))
da = np.array(kp_kd_v_list)


x_=da[:,0]
y_=da[:,1]
z_=da[:,2]
N=int(len(z_)**.5)
x = x_.reshape(N,N)
y = y_.reshape(N,N)
z = z_.reshape(N,N)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot_surface(x,y,z)
#ipdb.set_trace()
argmax = np.argmax(z_)
ax.scatter(x_[argmax], y_[argmax], z_[argmax],c='red')
ax.set_xlabel('kp')
ax.set_ylabel('kd')
ax.set_title('Value function f(kp,kd)')
plt.show()
      
