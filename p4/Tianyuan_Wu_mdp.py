import sys
import numpy as np
from copy import deepcopy


class node:
	'''
	This class is for book-keeping each state in the MDP.

	self.util is the utility of this state
	self.reward is R(s), the reward of getting to this state
	self.isWall is a flag that indicates whether this cell is occupied by wall
	self.isTerminal is a flag that indicates whether this state is a terminal state.

	'''
	def __init__(self, util = 0, reward=-0.04, isWall=False, isTerminal=False):
		self.util = util
		self.reward = reward
		self.isWall = isWall
		self.isTerminal = isTerminal

class grid:
	'''
	This class represents our MDP problem. It consists of

	- Description of the environment
	- Method to print the grid
	- Method to do Value Iteration (you need to implement)

	'''

	'''DO NOT EDIT THIS'''
	def __init__(self, gridfile):
		'''
		This init function helps you read and parse the environment file.
		And it maintains the below data members:

		self.gamma : discount factor
		self.living_cost : "cost of living", a negative reward
		self.mprobs : transition probabilities
		self.grid : a rectangle (defined by nrows-by-ncols) that represents
					the environment
		'''
		line = open(gridfile, 'r').readlines()
		self.gamma = float(line[0])
		self.living_cost = float(line[1])
		mprobs = line[2].split()
		self.mprobs = [float(n) for n in mprobs]
		self.grid = []
		for row in line[3:]:
			gridrow = []
			splitrow = row.split()
			if len(splitrow) == 0:
				continue
			for ch in row.split():
				if ch == '*':
					gridrow.append(node(reward=self.living_cost))
				elif ch == 'x':
					gridrow.append(node(isWall=True))
				else:
					try:
						utility = float(ch)
						gridrow.append(node(util=utility, isTerminal=True))
					except ValueError:
						print "Bad grid value"
						sys.exit()
			self.grid.append(gridrow)
		self.nrows = len(self.grid)
		assert(self.nrows > 0)
		self.ncols = len(self.grid[0])
		assert(self.ncols > 0)

	def printGrid(self):
		'''
		Print the grid
		'''
		printstr = ''
		for row in self.grid:
			for c in row:
				if not c.isWall:
					printstr += str(round(c.util, 3)) + ' '
				else:
					printstr += 'x '
			printstr += '\n'
		print printstr


	'''
		TODO:
	'''
	def doValueIteration(self, epsilon=0.00001):
		'''
		This function modifies the utilities of each cell in the grid.

		Input:
			epsilon : the maximum error allowed in the utility of any state
			self: this MDP problem

		Output:
			No need to return anything. You just need to modify the utility for
			each cell (state).
		'''
		r, c = np.shape(self.grid)[0], np.shape(self.grid)[1]
		updateU = np.zeros(r*c).reshape(r, c)
		for i in range(r):
			for j in range(c):
				if self.grid[i][j].isTerminal:
					if self.grid[i][j].util == 1: updateU[i][j] = 1
					elif self.grid[i][j].util == -1: updateU[i][j] = -1
		d = 1
		if self.gamma == 1: restriction = epsilon
		else: restriction = epsilon*(1-self.gamma)/self.gamma

		while d >= restriction:
			d = 0

			for i in range(r):
				for j in range(c):
					if not self.grid[i][j].isTerminal and not self.grid[i][j].isWall:
						northU, southU, westU, eastU = 0 , 0, 0, 0
						if i == 0:
							northU += self.mprobs[0]*self.grid[i][j].util
							westU += self.mprobs[1]*self.grid[i][j].util
							southU += self.mprobs[2]*self.grid[i][j].util
							eastU += self.mprobs[3]*self.grid[i][j].util
						elif self.grid[i-1][j].isWall:
							northU += self.mprobs[0]*self.grid[i][j].util
							westU += self.mprobs[1]*self.grid[i][j].util
							southU += self.mprobs[2]*self.grid[i][j].util
							eastU += self.mprobs[3]*self.grid[i][j].util
						else:
							northU += self.mprobs[0]*self.grid[i-1][j].util
							westU += self.mprobs[1]*self.grid[i-1][j].util
							southU += self.mprobs[2]*self.grid[i-1][j].util
							eastU += self.mprobs[3]*self.grid[i-1][j].util

						if j == 0:
							westU += self.mprobs[0]*self.grid[i][j].util
							southU += self.mprobs[1]*self.grid[i][j].util
							eastU += self.mprobs[2]*self.grid[i][j].util
							northU += self.mprobs[3]*self.grid[i][j].util
						elif self.grid[i][j-1].isWall:
							westU += self.mprobs[0]*self.grid[i][j].util
							southU += self.mprobs[1]*self.grid[i][j].util
							eastU += self.mprobs[2]*self.grid[i][j].util
							northU += self.mprobs[3]*self.grid[i][j].util
						else:
							westU += self.mprobs[0]*self.grid[i][j-1].util
							southU += self.mprobs[1]*self.grid[i][j-1].util
							eastU += self.mprobs[2]*self.grid[i][j-1].util
							northU += self.mprobs[3]*self.grid[i][j-1].util

						if i == r-1:
							southU += self.mprobs[0]*self.grid[i][j].util
							eastU += self.mprobs[1]*self.grid[i][j].util
							northU += self.mprobs[2]*self.grid[i][j].util
							westU += self.mprobs[3]*self.grid[i][j].util
						elif self.grid[i+1][j].isWall:
							southU += self.mprobs[0]*self.grid[i][j].util
							eastU += self.mprobs[1]*self.grid[i][j].util
							northU += self.mprobs[2]*self.grid[i][j].util
							westU += self.mprobs[3]*self.grid[i][j].util
						else:
							southU += self.mprobs[0]*self.grid[i+1][j].util
							eastU += self.mprobs[1]*self.grid[i+1][j].util
							northU += self.mprobs[2]*self.grid[i+1][j].util
							westU += self.mprobs[3]*self.grid[i+1][j].util

						if j == c-1:
							eastU += self.mprobs[0]*self.grid[i][j].util
							northU += self.mprobs[1]*self.grid[i][j].util
							westU += self.mprobs[2]*self.grid[i][j].util
							southU += self.mprobs[3]*self.grid[i][j].util
						elif self.grid[i][j+1].isWall:
							eastU += self.mprobs[0]*self.grid[i][j].util
							northU += self.mprobs[1]*self.grid[i][j].util
							westU += self.mprobs[2]*self.grid[i][j].util
							southU += self.mprobs[3]*self.grid[i][j].util
						else:
							eastU += self.mprobs[0]*self.grid[i][j+1].util
							northU += self.mprobs[1]*self.grid[i][j+1].util
							westU += self.mprobs[2]*self.grid[i][j+1].util
							southU += self.mprobs[3]*self.grid[i][j+1].util
						updateU[i][j] = self.grid[i][j].reward + self.gamma*max(northU, southU, westU, eastU)
						#print 'updated u at', i, j, 'is', updateU[i][j]
						#print 'original u at', i, j, 'is', self.grid[i][j].util
						if d < abs(self.grid[i][j].util - updateU[i][j]):
							d = abs(self.grid[i][j].util - updateU[i][j])
			for i in range(r):
				for j in range(c):
					self.grid[i][j].util = updateU[i][j]












if __name__=='__main__':
	assert(len(sys.argv)>1)
	g = grid(sys.argv[1])
	g.printGrid()
	g.doValueIteration()
	g.printGrid()
