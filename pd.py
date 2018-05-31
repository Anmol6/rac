import numpy as np
from env import mControl


ob_pos = 10
goal_pos = 20
ob_vel = -100
mass = 10
render = True
env = mControl(goal_pos=goal_pos, ob_pos=ob_pos, ob_vel=ob_vel, render=render)

ctrl_loops = 1000

def ctrl_fn(x, v, kp=60, kd=35):
    return -kp*x -kd*v

for i in range(ctrl_loops):
    x_ctrl = ob_pos-goal_pos
    v_ctrl = ob_vel 
    state,_,_,_ = env.step(ctrl_fn(x_ctrl,v_ctrl))
    ob_pos = state[0]
    ob_vel = state[1]
       
