from rrt_gym import RRT
from math import pi

start = [pi/2, pi/4, pi/2, pi/4, pi/2]
goal = [pi/8, 3*pi/4, pi, 0.9*pi, 1.5*pi]

planner = RRT("map1.txt", start, goal)

state = planner.reset()

done = False

cnt = 0

while not done:
    action = 1 * (cnt % 2 == 0) # alternating action now
    state, reward, done, _ = planner.step(action)
    print(reward)
    cnt += 1

plan = planner.plan
print(cnt)
print("Final Cost: {}, Num of nodes generated: {}, Path Length: {}".format(planner.calc_cost(), planner.num_of_nodes, len(plan)))