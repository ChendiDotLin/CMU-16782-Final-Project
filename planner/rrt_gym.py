# rewrite of rrt_planner in openai gym style

import numpy as np
import enum
import math
import random
import time


class Node:
    def __init__(self, numofDOFs):
        self.arm_anglesV_rad = [0] * numofDOFs
        self.parent = None


class ExtendStatus():
    REACHED = 1
    ADVANCED = 2
    TRAPPED = 3


class bresenham_param_t:
    def __init__(self):
        self.X1 = 0
        self.Y1 = 0
        self.X2 = 0
        self.Y2 = 0
        self.Increment = 0
        self.UsingYIndex = 0
        self.DeltaX = 0
        self.DeltaY = 0
        self.DTerm = 0
        self.IncrE = 0
        self.IncrNE = 0
        self.XIndex = 0
        self.YIndex = 0
        self.Flipped = 0


class RRT(object):

    def __init__(self, path_to_map, start, goal):
        self.env = path_to_map
        self.start = start
        self.goal = goal

        self.map = np.loadtxt(self.env)
        self.x_size = self.map.shape[0]
        self.y_size = self.map.shape[1]
        self.numofDOFs = len(start)

        if not self.IsValidArmConfiguration(start, self.numofDOFs, self.map, self.x_size, self.y_size):
            raise NameError("arm start configuration is invalid. Please change another one.")
        if not self.IsValidArmConfiguration(goal, self.numofDOFs, self.map, self.x_size, self.y_size):
            raise NameError("arm goal configuration is invalid. Please change another one.")

        self.step_size = math.pi / 90
        self.interpolate_times = 10
        self.max_expand_times = 8000

        # used in state representation
        self.neighboring_distance = 10

        self.reset()

    def reset(self):
        self.q_start = Node(self.numofDOFs)
        self.q_goal = Node(self.numofDOFs)

        for i in range(self.numofDOFs):
            self.q_start.arm_anglesV_rad[i] = self.start[i]
            self.q_goal.arm_anglesV_rad[i] = self.goal[i]

        self.node_list_forward = []
        self.node_list_backward = []
        self.planned_path = []
        self.node_list_forward.append(self.q_start)
        self.node_list_backward.append(self.q_goal)

        self.cal_cost_flag = True
        self.done = False
        self.goal_reached = False

        self.q_new = [0,] * self.numofDOFs

        self.expand_times = 0

        return self.state


    # state representation for RL:
    # [goal, q_new, N_neighbors_forward, N_neighbors_backward,
    # D_nearest_neighbor_forward, D_nearest_neightbor_backward,
    # Length_forward, Length_backward]
    @property
    def state(self):
        state = []
        state.extend(self.goal)
        state.extend(self.q_new)

        # number of neighboring pnts within self.neighboring_distance in two trees
        N_neighbors_forward = 0
        N_neighbors_backward = 0
        # distance to closest pnt in two trees
        D_nearest_neighbor_forward = 100
        D_nearest_neightbor_backward = 100

        for i in range(len(self.node_list_forward)):
            node = self.node_list_forward[i].arm_anglesV_rad
            dis = np.linalg.norm(np.array(node) - np.array(self.q_new))
            if dis < self.neighboring_distance:
                N_neighbors_forward += 1
            D_nearest_neighbor_forward = min(D_nearest_neighbor_forward, dis)

        for i in range(len(self.node_list_backward)):
            node = self.node_list_backward[i].arm_anglesV_rad
            dis = np.linalg.norm(np.array(node) - np.array(self.q_new))
            if dis < self.neighboring_distance:
                N_neighbors_backward += 1
            D_nearest_neightbor_backward = min(D_nearest_neightbor_backward, dis)

        state.extend([N_neighbors_forward, N_neighbors_backward, D_nearest_neighbor_forward, D_nearest_neightbor_backward])

        # also append the length of the forward/backward tree
        state.append(len(self.node_list_forward))
        state.append(len(self.node_list_backward))
        return state

    # action should be [0, 1]. 0: expand from the start tree; 1: expand from the goal tree
    def step(self, action):
        assert (action == 0 or action == 1)

        q_rand = self.getRandomNode(self.numofDOFs)

        self.q_new = q_rand.arm_anglesV_rad

        q_new = Node(self.numofDOFs)
        q_new_forward = Node(self.numofDOFs)
        q_new_backward = Node(self.numofDOFs)

        connect = False

        # start tree
        if action == 0:
            result = self.extend(q_rand, q_new, self.node_list_forward, self.step_size, self.interpolate_times,
                                 self.numofDOFs, self.map, self.x_size,self.y_size, connect)

            if (result != ExtendStatus.TRAPPED):
                connect = True
                result_1 = ExtendStatus()

                while True:
                    result_1 = self.extend(q_new, q_new_backward, self.node_list_backward, self.step_size, self.interpolate_times,
                                      self.numofDOFs, self.map, self.x_size, self.y_size, connect)
                    if (result_1 != ExtendStatus.ADVANCED):
                        break

                if (self.reached(q_new, q_new_backward, self.numofDOFs)):
                    # q_new.parent = q_new_backward
                    self.node_list_backward.append(q_new)

                    while True:
                        if (q_new.parent != None):
                            self.planned_path.append(q_new)
                            q_new = q_new.parent
                            if (q_new.parent == None):
                                self.planned_path.append(q_new)
                        else:
                            break

                    self.planned_path.reverse()

                    while True:
                        if (q_new_backward.parent != None):
                            self.planned_path.append(q_new_backward)
                            q_new_backward = q_new_backward.parent
                            if (q_new_backward.parent == None):
                                self.planned_path.append(q_new_backward)
                        else:
                            break
                    #break
                    self.done = True
                    self.goal_reached = True
        else:
            result = self.extend(q_rand, q_new, self.node_list_backward, self.step_size, self.interpolate_times,
                                 self.numofDOFs, self.map, self.x_size, self.y_size, connect)

            if result != ExtendStatus.TRAPPED:
                connect = True
                result_1 = ExtendStatus()

                while True:
                    result_1 = self.extend(q_new, q_new_forward, self.node_list_forward, self.step_size, self.interpolate_times, self.numofDOFs,
                                      self.map, self.x_size, self.y_size, connect)
                    if result_1 != ExtendStatus.ADVANCED:
                        break

                if self.reached(q_new, q_new_forward, self.numofDOFs):
                    # q_new.parent = q_new_forward
                    self.node_list_forward.append(q_new)

                    while True:
                        if q_new_forward.parent != None:
                            self.planned_path.append(q_new_forward)
                            q_new_forward = q_new_forward.parent
                            if q_new_forward.parent == None:
                                self.planned_path.append(q_new_forward)
                        else:
                            break

                    self.planned_path.reverse()

                    while True:
                        if q_new.parent != None:
                            self.planned_path.append(q_new)
                            q_new = q_new.parent
                            if q_new.parent == None:
                                self.planned_path.append(q_new)
                        else:
                            break
                    self.done = True
                    self.goal_reached = True

        # end_time = time.time()
        # planning_time = end_time - start_time
        # if planning_time > upper_limit_time:
        #     print("We did not find the path under the upper limit time ", upper_limit_time)
        #     cal_cost_flag = False
        #     break
        self.expand_times += 1
        if self.expand_times >= self.max_expand_times:
            self.done = True

        # calc reward
        reward = (1000 * self.goal_reached) / self.expand_times

        # current reward function: binary reward of whether goal is reached
        return self.state, reward, self.done, "goal reached" if (self.goal_reached == True) else "failed"

    def calc_cost(self):
        cost = 0
        for i in range(len(self.planned_path) - 1):
            dist = 0
            for j in range(self.numofDOFs):
                dist = dist + pow(self.planned_path[i + 1].arm_anglesV_rad[j] - self.planned_path[i].arm_anglesV_rad[j], 2)

            dist = np.sqrt(dist)
            cost = cost + dist

        print("The cost is ", cost)
        return cost

    @property
    def plan(self):
        plan = [i.arm_anglesV_rad for i in self.planned_path]
        return plan

    @property
    def num_of_nodes(self):
        return len(self.node_list_forward) + len(self.node_list_backward)

    def GETMAPINDEX(self, x, y, x_size, y_size):
        return y * x_size + x

    def ContXY2Cell(self, x, y, pX, pY, x_size, y_size):
        cellsize = 1.0
        pX = int(x / cellsize)
        if x < 0:
            pX = 0
        if pX >= x_size:
            pX = x_size - 1
        pY = int(y / cellsize)
        if y < 0:
            pY = 0
        if pY >= y_size:
            pY = y_size - 1
        return pX, pY

    def get_bresenham_parameters(self, p1x, p1y, p2x, p2y, params):
        params.UsingYIndex = 0

        if ((p1x == p2x) or abs((p2y - p1y) / (p2x - p1x)) > 1):
            params.UsingYIndex += 1

        if (params.UsingYIndex):
            params.Y1 = p1x
            params.X1 = p1y
            params.Y2 = p2x
            params.X2 = p2y
        else:
            params.X1 = p1x
            params.Y1 = p1y
            params.X2 = p2x
            params.Y2 = p2y

        if ((p2x - p1x) * (p2y - p1y) < 0):
            params.Flipped = 1
            params.Y1 = -params.Y1
            params.Y2 = -params.Y2
        else:
            params.Flipped = 0

        if (params.X2 > params.X1):
            params.Increment = 1
        else:
            params.Increment = -1

        params.DeltaX = params.X2 - params.X1
        params.DeltaY = params.Y2 - params.Y1

        params.IncrE = 2 * params.DeltaY * params.Increment
        params.IncrNE = 2 * (params.DeltaY - params.DeltaX) * params.Increment
        params.DTerm = (2 * params.DeltaY - params.DeltaX) * params.Increment

        params.XIndex = params.X1
        params.YIndex = params.Y1

    def get_current_point(self, params, x, y):
        if (params.UsingYIndex):
            y = params.XIndex
            x = params.YIndex
            if (params.Flipped):
                x = -x
        else:
            x = params.XIndex
            y = params.YIndex
            if (params.Flipped):
                y = -y
        return x, y

    def get_next_point(self, params):
        if (params.XIndex == params.X2):
            return False
        params.XIndex += params.Increment
        if (params.DTerm < 0 or (params.Increment < 0 and params.DTerm <= 0)):
            params.DTerm += params.IncrE
        else:
            params.DTerm += params.IncrNE
            params.YIndex += params.Increment
        return True

    def IsValidLineSegment(self, x0, y0, x1, y1, map, x_size, y_size):
        params = bresenham_param_t()
        nX = 0
        nY = 0
        nX0 = 0
        nY0 = 0
        nX1 = 0
        nY1 = 0

        if (x0 < 0 or x0 >= x_size or x1 < 0 or x1 >= x_size or y0 < 0 or y0 >= y_size or y1 < 0 or y1 >= y_size):
            return False

        nX0, nY0 = self.ContXY2Cell(x0, y0, nX0, nY0, x_size, y_size)
        nX1, nY1 = self.ContXY2Cell(x1, y1, nX1, nY1, x_size, y_size)

        self.get_bresenham_parameters(nX0, nY0, nX1, nY1, params)

        while True:
            nX, nY = self.get_current_point(params, nX, nY)
            if (map[nX][nY] == 1):
                return False
            if not self.get_next_point(params):
                break

        return True

    def IsValidArmConfiguration(self, angles, numofDOFs, map, x_size, y_size):
        x0 = 0.0
        y0 = 0.0
        x1 = 0.0
        y1 = 0.0

        x1 = x_size / 2.0
        for i in range(numofDOFs):
            x0 = x1
            y0 = y1
            x1 = x0 + 10 * math.cos(2 * math.pi - angles[i])
            y1 = y0 - 10 * math.sin(2 * math.pi - angles[i])

            if not self.IsValidLineSegment(x0, y0, x1, y1, map, x_size, y_size):
                return False

        return True

    def getRandomNode(self, numofDOFs):
        ret = Node(numofDOFs)
        for j in range(numofDOFs):
            ret.arm_anglesV_rad[j] = random.random() * 2 * math.pi
        return ret

    def findNearest(self, q_rand, node_list, numofDOFs):
        min_dist_index = 0
        min_dist = math.inf
        for i in range(len(node_list)):
            dist = 0
            for j in range(numofDOFs):
                dist = dist + pow(node_list[i].arm_anglesV_rad[j] - q_rand.arm_anglesV_rad[j], 2)
            dist = np.sqrt(dist)
            if dist < min_dist:
                min_dist = dist
                min_dist_index = i
        # print(min_dist_index)
        # print(len(node_list))
        q_near = node_list[min_dist_index]
        return q_near

    def newConfig(self, q_rand, q_near, q_new, step_size, interpolate_times, numofDOFs, map, x_size, y_size, connect):
        distance = 0
        for j in range(numofDOFs):
            if (distance < abs(q_rand.arm_anglesV_rad[j] - q_near.arm_anglesV_rad[j])):
                distance = abs(q_rand.arm_anglesV_rad[j] - q_near.arm_anglesV_rad[j])
        numofSamples = int(distance / step_size)

        flag = False
        new_angles = Node(numofDOFs)

        if connect == True:
            for i in range(numofSamples):
                for j in range(numofDOFs):
                    new_angles.arm_anglesV_rad[j] = q_near.arm_anglesV_rad[j] + (i + 1) / numofSamples * (
                            q_rand.arm_anglesV_rad[j] - q_near.arm_anglesV_rad[j])
                if (self.IsValidArmConfiguration(new_angles.arm_anglesV_rad, numofDOFs, map, x_size, y_size)):
                    for k in range(numofDOFs):
                        q_new.arm_anglesV_rad[k] = new_angles.arm_anglesV_rad[k]
                    flag = True
                else:
                    break
        else:
            if numofSamples > interpolate_times:
                for i in range(interpolate_times):
                    for j in range(numofDOFs):
                        new_angles.arm_anglesV_rad[j] = q_near.arm_anglesV_rad[j] + (i + 1) / numofSamples * (
                                q_rand.arm_anglesV_rad[j] - q_near.arm_anglesV_rad[j])
                    if (self.IsValidArmConfiguration(new_angles.arm_anglesV_rad, numofDOFs, map, x_size, y_size)):
                        for k in range(numofDOFs):
                            q_new.arm_anglesV_rad[k] = new_angles.arm_anglesV_rad[k]
                        flag = True
                    else:
                        break
            else:
                for i in range(numofSamples):
                    for j in range(numofDOFs):
                        new_angles.arm_anglesV_rad[j] = q_near.arm_anglesV_rad[j] + (i + 1) / numofSamples * (
                                q_rand.arm_anglesV_rad[j] - q_near.arm_anglesV_rad[j])
                    if (self.IsValidArmConfiguration(new_angles.arm_anglesV_rad, numofDOFs, map, x_size, y_size)):
                        for k in range(numofDOFs):
                            q_new.arm_anglesV_rad[k] = new_angles.arm_anglesV_rad[k]
                        flag = True
                    else:
                        break

        if flag == True:
            return True, q_new
        else:
            return False, q_new

    def reached(self, q_goal, q_new, numofDOFs):
        dist = 0
        for j in range(numofDOFs):
            dist = dist + pow(q_goal.arm_anglesV_rad[j] - q_new.arm_anglesV_rad[j], 2)

        dist = np.sqrt(dist)
        if dist < 0.01:
            return True
        else:
            return False

    def extend(self, q, q_new, node_list, step_size, interpolate_times, numofDOFs, map, x_size, y_size, connect):
        q_near = self.findNearest(q, node_list, numofDOFs)
        flag, q_new = self.newConfig(q, q_near, q_new, step_size, interpolate_times, numofDOFs, map, x_size, y_size, connect)
        if (flag == True):
            q_new.parent = q_near
            node_list.append(q_new)

            if (self.reached(q, q_new, numofDOFs)):
                return ExtendStatus.REACHED
            else:
                return ExtendStatus.ADVANCED
        else:
            return ExtendStatus.TRAPPED