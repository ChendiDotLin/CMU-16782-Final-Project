from rrt_gym import RRT
from math import pi

start = [0, 0]#[pi/10, pi/4, pi/2]
goal = [1, 1]#[pi/8, 3*pi/4, pi]

planner = RRT("map2.txt", start, goal)

state = planner.reset()

done = False

cnt = 0

import numpy as np
np.random.seed(1)

while not done:
    action = 1 * (cnt % 2 == 0) # alternating action now
    state, reward, done, _ = planner.step(action)
    #print(reward)
    cnt += 1
    if cnt % 1000 == 0:
        print(cnt)

plan = planner.plan
print(cnt)
print("Final reward: {}, Num of nodes generated: {}, Path Length: {}".format(planner.calc_cost(), planner.num_of_nodes, len(plan)))