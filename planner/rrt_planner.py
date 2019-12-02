import numpy as np
import enum
import math

#math.pi

class Node:
	def __init__(self, numofDOFs):
		self.arm_anglesV_rad = np.zeros(numofDOFs)

class ExtendStatus(enum.Enum):
	REACHED = 1
	ADVANCED = 2
	TRAPPED = 3

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

def getRandomNode()








def planner(env,start,goal):
	print(env[0])
	map = np.loadtxt(env)
	print('start is ', start)






if __name__ == "__main__":
    pass