import numpy as np
import enum
import math
import random
import time
import bandits

pi = 3.1416926535897

class Node:
	def __init__(self, numofDOFs):
		self.arm_anglesV_rad = [0]*numofDOFs
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

def GETMAPINDEX(x, y, x_size, y_size):
	return y*x_size + x

def ContXY2Cell(x, y, pX, pY, x_size, y_size):
	cellsize = 1.0
	pX = int(x/cellsize)
	if x < 0:
		pX = 0
	if pX >= x_size:
		pX = x_size - 1
	pY = int(y/cellsize)
	if y < 0:
		pY = 0
	if pY >= y_size:
		pY = y_size - 1
	return pX, pY


def get_bresenham_parameters(p1x, p1y, p2x, p2y, params):
	params.UsingYIndex = 0

	if ((p1x == p2x) or abs((p2y-p1y)/(p2x-p1x)) > 1):
		params.UsingYIndex+=1

	if(params.UsingYIndex):
		params.Y1 = p1x
		params.X1 = p1y
		params.Y2 = p2x
		params.X2 = p2y
	else:
		params.X1 = p1x
		params.Y1 = p1y
		params.X2 = p2x
		params.Y2 = p2y

	if((p2x-p1x)*(p2y-p1y)<0):
		params.Flipped = 1
		params.Y1 = -params.Y1
		params.Y2 = -params.Y2
	else:
		params.Flipped = 0

	if(params.X2 > params.X1):
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

def get_current_point(params, x, y):
	if(params.UsingYIndex):
		y = params.XIndex
		x = params.YIndex
		if(params.Flipped):
			x = -x
	else:
		x = params.XIndex
		y = params.YIndex
		if(params.Flipped):
			y = -y
	return x,y

def get_next_point(params):
	if(params.XIndex == params.X2):
		return False
	params.XIndex += params.Increment
	if(params.DTerm < 0 or (params.Increment < 0 and params.DTerm <=0)):
		params.DTerm += params.IncrE
	else:
		params.DTerm += params.IncrNE
		params.YIndex += params.Increment
	return True

def IsValidLineSegment(x0, y0, x1, y1, map, x_size, y_size):
	params = bresenham_param_t()
	nX = 0
	nY = 0
	nX0 = 0
	nY0 = 0
	nX1 = 0
	nY1 = 0

	if(x0 < 0 or x0 >= x_size or x1 < 0 or x1 >= x_size or y0 < 0 or y0 >= y_size or y1 < 0 or y1 >= y_size):
		return False

	nX0, nY0 = ContXY2Cell(x0, y0, nX0, nY0, x_size, y_size)
	nX1, nY1 = ContXY2Cell(x1, y1, nX1, nY1, x_size, y_size)

	get_bresenham_parameters(nX0, nY0, nX1, nY1, params)

	while True:
		nX, nY = get_current_point(params, nX, nY)
		if(map[nX][nY] == 1):
			return False
		if not get_next_point(params):
			break

	return True

def IsValidArmConfiguration(angles, numofDOFs, map, x_size, y_size):
	x0 = 0.0
	y0 = 0.0
	x1 = 0.0
	y1 = 0.0

	x1 = x_size/2.0
	for i in range(numofDOFs):
		x0 = x1
		y0 = y1
		x1 = x0 + 10 * math.cos(2 * math.pi - angles[i])
		y1 = y0 - 10 * math.sin(2 * math.pi - angles[i])

		if not IsValidLineSegment(x0, y0, x1, y1, map, x_size, y_size):
			return False

	return True


def getRandomNode(numofDOFs):
	ret = Node(numofDOFs)
	for j in range(numofDOFs):
		ret.arm_anglesV_rad[j] = random.random() * 2 * math.pi
	return ret

def findNearest(q_rand, node_list, numofDOFs):
	min_dist_index = 0
	min_dist = np.Inf
	for i in range(len(node_list)):
		dist = 0
		for j in range(numofDOFs):
			dist = dist + pow(node_list[i].arm_anglesV_rad[j] - q_rand.arm_anglesV_rad[j], 2)
		dist = np.sqrt(dist)
		if dist < min_dist:
			min_dist = dist
			min_dist_index = i
	#print(min_dist_index)
	#print(len(node_list))
	q_near = node_list[min_dist_index]
	return q_near

def newConfig(q_rand, q_near, q_new, step_size, interpolate_times, numofDOFs, map, x_size, y_size, connect):
	distance = 0
	for j in range(numofDOFs):
		if(distance < abs(q_rand.arm_anglesV_rad[j] - q_near.arm_anglesV_rad[j])):
			distance = abs(q_rand.arm_anglesV_rad[j] - q_near.arm_anglesV_rad[j])
	numofSamples = int (distance/step_size)

	flag = False
	new_angles = Node(numofDOFs)

	if connect == True:
		for i in range(numofSamples):
			for j in range(numofDOFs):
				new_angles.arm_anglesV_rad[j] = q_near.arm_anglesV_rad[j] + (i+1)/numofSamples * (q_rand.arm_anglesV_rad[j] - q_near.arm_anglesV_rad[j])
			if(IsValidArmConfiguration(new_angles.arm_anglesV_rad, numofDOFs, map, x_size, y_size)):
				for k in range(numofDOFs):
					q_new.arm_anglesV_rad[k] = new_angles.arm_anglesV_rad[k]
				flag = True
			else:
				break
	else:
		if numofSamples > interpolate_times:
			for i in range(interpolate_times):
				for j in range(numofDOFs):
					new_angles.arm_anglesV_rad[j] = q_near.arm_anglesV_rad[j] + (i+1)/numofSamples * (q_rand.arm_anglesV_rad[j] - q_near.arm_anglesV_rad[j])
				if(IsValidArmConfiguration(new_angles.arm_anglesV_rad, numofDOFs, map, x_size, y_size)):
					for k in range(numofDOFs):
						q_new.arm_anglesV_rad[k] = new_angles.arm_anglesV_rad[k]
					flag = True
				else:
					break
		else:
			for i in range(numofSamples):
				for j in range(numofDOFs):
					new_angles.arm_anglesV_rad[j] = q_near.arm_anglesV_rad[j] + (i+1)/numofSamples * (q_rand.arm_anglesV_rad[j] - q_near.arm_anglesV_rad[j])
				if(IsValidArmConfiguration(new_angles.arm_anglesV_rad, numofDOFs, map, x_size, y_size)):
					for k in range(numofDOFs):
						q_new.arm_anglesV_rad[k] = new_angles.arm_anglesV_rad[k]
					flag = True
				else:
					break

	if flag == True:
		return True, q_new
	else:
		return False, q_new

def reached(q_goal, q_new, numofDOFs):
	dist = 0
	for j in range(numofDOFs):
		dist = dist + pow(q_goal.arm_anglesV_rad[j] - q_new.arm_anglesV_rad[j], 2)

	dist = np.sqrt(dist)
	if dist < 0.01:
		return True
	else:
		return False

def extend(q, q_new, node_list, step_size, interpolate_times, numofDOFs, map, x_size, y_size, connect):
	q_near = findNearest(q, node_list, numofDOFs)
	flag, q_new = newConfig(q, q_near, q_new, step_size, interpolate_times, numofDOFs, map, x_size, y_size, connect)
	if(flag == True):
		q_new.parent = q_near
		node_list.append(q_new)

		if(reached(q, q_new, numofDOFs)):
			return ExtendStatus.REACHED
		else:
			return ExtendStatus.ADVANCED
	else:
		return ExtendStatus.TRAPPED


def planner(env,start,goal):	 
	#print(env[0])
	print(start)
	print(goal)
	random.seed(1)

	map = np.loadtxt(env)
	x_size = map.shape[0]
	y_size = map.shape[1]
	numofDOFs = len(start)
	# print(numofDOFs)
	# print(numofDOFs)
	if not IsValidArmConfiguration(start, numofDOFs, map, x_size, y_size):
		print("arm start configuration is invalid. Please change another one.")
		return
	if not IsValidArmConfiguration(goal, numofDOFs, map, x_size, y_size):
		print("arm goal configuration is invalid. Please change another one.")
		return

	start_time = time.time()
	#end_time = time.time()
	#print end_time - start_time
	upper_limit_time = 300

	q_start = Node(numofDOFs)
	q_goal = Node(numofDOFs)

	for i in range(numofDOFs):
		q_start.arm_anglesV_rad[i] = start[i]
		q_goal.arm_anglesV_rad[i] = goal[i]

	node_list_forward = []
	node_list_backward = []
	planned_path = []
	node_list_forward.append(q_start)
	node_list_backward.append(q_goal)

	step_size = math.pi/90
	interpolate_times = 10
	max_expand_times = 100000

	cal_cost_flag = True
	# policy = bandits.Policy()
	policy = bandits.policyUCB(2)
	# or 
	# policy = bandits.policyDTS(2)
	# print(ucb.nbActions)

	# dts = bandits.policyDTS(policy)
    
	for k in range(max_expand_times):
		q_rand = getRandomNode(numofDOFs)

		q_new = Node(numofDOFs)
		q_new_forward = Node(numofDOFs)
		q_new_backward = Node(numofDOFs)

		connect = False

		# if (k>1):
		# 	print(policy.action)
		if (policy.decision() == 0):
			# print(policy.action)
		# if(k%2==0):
			result = extend(q_rand, q_new, node_list_forward, step_size, interpolate_times, numofDOFs, map, x_size, y_size, connect)
			if (result==ExtendStatus.REACHED):
				reward  = 0.1
			elif(result==ExtendStatus.ADVANCED):
				reward = 0.4
			else:
				reward = 0.9

			policy.getReward(reward)
			
			if(result != ExtendStatus.TRAPPED):
				connect = True
				result_1 = ExtendStatus()

				while True:
					result_1 = extend(q_new, q_new_backward, node_list_backward, step_size, interpolate_times, numofDOFs, map, x_size, y_size,connect)
					if(result_1 != ExtendStatus.ADVANCED):
						break

				if(reached(q_new, q_new_backward, numofDOFs)):
					#q_new.parent = q_new_backward
					node_list_backward.append(q_new)
                
					while True:
						if(q_new.parent != None):
							planned_path.append(q_new)
							q_new = q_new.parent
							if(q_new.parent == None):
								planned_path.append(q_new)
						else:
							break

					planned_path.reverse()

					while True:
						if(q_new_backward.parent != None):
							planned_path.append(q_new_backward)
							q_new_backward = q_new_backward.parent
							if(q_new_backward.parent == None):
								planned_path.append(q_new_backward)
						else:
							break
					break

		else:
			result = extend(q_rand, q_new, node_list_backward, step_size, interpolate_times, numofDOFs, map, x_size, y_size, connect)
			if (result==ExtendStatus.REACHED):
				reward  = 0.1
			elif(result==ExtendStatus.ADVANCED):
				reward = 0.4
			else:
				reward = 0.9

			policy.getReward(reward)


			if(result != ExtendStatus.TRAPPED):
				connect = True
				result_1 = ExtendStatus()

				while True:
					result_1 = extend(q_new, q_new_forward, node_list_forward, step_size, interpolate_times, numofDOFs, map, x_size, y_size, connect)
					if(result_1 != ExtendStatus.ADVANCED):
						break

				if(reached(q_new, q_new_forward, numofDOFs)):
					#q_new.parent = q_new_forward
					node_list_forward.append(q_new)

					while True:
						if(q_new_forward.parent != None):
							planned_path.append(q_new_forward)
							q_new_forward = q_new_forward.parent
							if(q_new_forward.parent == None):
								planned_path.append(q_new_forward)
						else:
							break

					planned_path.reverse()

					while True:
						if(q_new.parent != None):
							planned_path.append(q_new)
							q_new = q_new.parent
							if(q_new.parent == None):
								planned_path.append(q_new)
						else:
							break
					break

		end_time = time.time()
		planning_time = end_time - start_time
		if planning_time > upper_limit_time:
			print("We did not find the path under the upper limit time ", upper_limit_time)
			cal_cost_flag = False
			break

	if(cal_cost_flag == True):
		cost = 0
		for i in range(len(planned_path)-1):
			dist = 0
			for j in range(numofDOFs):
				dist = dist + pow(planned_path[i+1].arm_anglesV_rad[j] - planned_path[i].arm_anglesV_rad[j], 2)

			dist = np.sqrt(dist)
			cost = cost + dist

		print("The cost is ", cost)
		end_time = time.time()
		planning_time = end_time - start_time
		print("We find the path and the planning time is ", planning_time)

	print("number of nodes generated ", len(node_list_forward) + len(node_list_backward))

	plan = [i.arm_anglesV_rad for i in planned_path]
	# print(plan)
	return plan,len(node_list_forward) + len(node_list_backward)



if __name__ == "__main__":
	res = 0
	for i in range(1):
		# plan,expansion = planner("map2.txt",[0,0],[1,1])
		# plan,expansion = planner("map2.txt"	,[0, 0, pi/2, pi/2, pi/2],[pi/8, 3*pi/4, pi, 0.9*pi, 1.5*pi])
		plan,expansion = planner("map2.txt"	,[pi/10, pi/4, pi/2],[pi/8, 3*pi/4, pi])
		

		res += expansion
	# res /=50
	print(res)
	print(len(plan))
