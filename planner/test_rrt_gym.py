from rrt_gym import RRT

planner = RRT("map2.txt", [0, 0], [1, 1])

state = planner.reset()

done = False

cnt = 0

while not done:
    action = 1 * (cnt % 2 == 0)
    state, reward, done, _ = planner.step(action)

plan = planner.plan
print("Final Cost: {}, Num of nodes generated: {}, Path Length: {}".format(planner.calc_cost(), planner.num_of_nodes, len(plan)))