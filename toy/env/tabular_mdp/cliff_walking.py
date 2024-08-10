import copy
import gym
from gym import spaces

class CliffWalkingEnv(gym.Env):

    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        self.x = 0
        self.y = self.nrow - 1 # 坐标系原点在左上角
        # self.P = self.createP()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(ncol * nrow)


    # def createP(self):
    #     P= [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]

    #     change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
    #     for i in range(self.nrow):
    #         for j in range(self.ncol):
    #             for a in range(4):
    #                 if i == self.nrow - 1 and j > 0:
    #                     P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
    #                     continue
                    
    #                 next_x = min(self.ncol - 1, max(0, self.x + change[a][0]))
    #                 next_y = min(self.nrow - 1, max(0, self.y + change[a][1]))
    #                 next_state = next_y * self.ncol + next_x
    #                 reward = -1
    #                 done = False

    #                 if next_y == self.nrow - 1 and next_x > 0:
    #                     done = True
    #                     if next_x != self.ncol - 1:
    #                         reward = -100
    #                 P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
    #     return P


    def step(self, action): 
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]] # 上，下，左，右
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            reward = 100   # 11可以，10不可以，(eps=0.2)；6可以，5不可以，(eps=0.1)；
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x