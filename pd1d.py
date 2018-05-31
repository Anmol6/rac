import numpy as np
from env import 1dControl #mport 1dControl


ob_pos = 10
goal_pos = 20
ob_vel = -10
mass = 10
render = True
env = 1dControl(goal_pos=goal_pos, ob_pos=ob_pos, ob_vel=ob_vel, render=render)

ctrl_loops = 1000

def ctrl_fn(x, v, kd=10, kv=5):
    return -kd*x -kv*v

for i in ctrl_loops:
    x_ctrl = ob_pos-goal_pos
    
    if ob_pos<goal_pos
        v_ctrl = -ob_vel
    state,_,_,_ = env.step(ctrl_fn(x_ctrl,v_ctrl))
    ob_pos = state[0]
    ob_vel = state[0]
       
