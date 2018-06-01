import numpy as np
import matplotlib.pyplot as plt
import ipdb
class mControl:
    def __init__(self, dt=0.001, goal_pos=0.0, ob_pos=1.0, ob_vel=0.0, mass=10, render=False):
        self.goal_pos = goal_pos
        self.dt = dt
        self.init_pos = ob_pos
        self.init_vel = ob_vel
        self.pos = ob_pos
        self.vel = ob_vel
        self.mass = mass
        self.itr=0
        self.k = 1.0
        self.eps = 1e-1#1e-3 #task is considered done when object is within epsilon of goal
        #Note state includes relative position of object w.r.t goal not absolute position
        self.done = False
        self.render = render
        
        if self.render:
            plt.ion() 
            self.fig=plt.figure()
            self.ax=self.fig.add_subplot(111)
            self.ax.set_ylabel('position (with goal at  ' + str(self.goal_pos)+')')
            self.ax.set_xlabel('iterations')
            plt.pause(0.01)
        pass
    
    def get_reward(self):
        #ipdb.set_trace()
        return -(self.pos-self.goal_pos)**2 - 1e-11*(self.action**2)
      
    def step(self, action):
        """ action is 1d force applied on mass """
        
        self.vel = self.vel + (action/self.mass) * self.dt
        self.pos = self.pos + self.vel*self.dt
        self.action = action 
        if self.render and self.itr%100==0:
            self._render()
        if abs(self.pos - self.goal_pos) < self.eps and self.itr>200:
            self.done=True
        self.itr+=1

        return np.array([self.pos, self.vel]).reshape(1,2), self.get_reward(), self.done, None

    def set_pos(self, val):
        self.goal_pos = val
        if self.render:
           self.ax.set_ylabel('goal_pos ' + str(val))
    def reset(self):
        self.pos = self.init_pos
        self.vel = self.init_vel
        self.itr=0
        self.done=False
        return np.array([self.pos,self.vel]).reshape(1,2)

    def _render(self):
        self.ax.scatter(self.itr, self.pos)
        self.ax.set_xlim(self.itr-50, self.itr+4)
        #self.ax.text(0.5,0.5, 'x_position: ' + str(self.pos) + '|' 'x_velocity: ' + str(self.vel))
        plt.show()
        plt.pause(0.01)
