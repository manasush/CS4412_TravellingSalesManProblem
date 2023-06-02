#CS4412
#Project 5: TSP with branch and bound
#Sushan Manandhar
#Professor Bodily
#Date:04/16/2021
# Citation:
# https://www.geeksforgeeks.org/traveling-salesman-problem-tsp-implementation/
# https://www.geeksforgeeks.org/travelling-salesman-problem-set-1/

#!/usr/bin/python3

from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
	def __init__(self, gui_view):
		self._scenario = None
		self.cities = None

	def setupWithScenario(self, scenario):
		self._scenario = scenario
		self.cities = self._scenario.getCities()

	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def defaultRandomTour(self, time_allowance=60.0):
		results = {}
		ncities = len(self.cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time() - start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation(ncities)
			route = []
			# Now build the route using the random permutation
			for i in range(ncities):
				route.append(self.cities[perm[i]])
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''


	def branchAndBound(self, time_allowance=60.0):

		# initialization
		results = {}
		maxQueueSize = 0
		totalStatesCreated = 0

		prunedStatesCount = 0

		pq = []
		heapq.heapify(pq)
		start_time = time.time()
		bssf = self.greedy()['soln']
		foundTour = True
		count = 0

		# set up state
		heapq.heappush(pq, self.initFirstState())
		totalStatesCreated += 1


		while len(pq) > 0 and time.time() - start_time < time_allowance:
			currentState = heapq.heappop(pq)
			if currentState.cost < bssf.cost:
				currentState.reduce()
				if currentState.cost < bssf.cost:
					if currentState.depth == len(self.cities):
						# update bssf
						bssf = TSPSolution(currentState.path)
						count += 1
					else:
						# generate children
						newStates = currentState.getChildren()
						totalStatesCreated += len(newStates)
						for state in newStates:
							heapq.heappush(pq, state)
						if len(pq) > maxQueueSize:
							maxQueueSize = len(pq)
				else:
					# prune
					prunedStatesCount += 1
			else:
				# prune
				prunedStatesCount += 1

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = maxQueueSize
		results['total'] = totalStatesCreated
		results['pruned'] = prunedStatesCount
		return results

	def initFirstState(self):
		cost = 0
		depth = 1
		cityIndex = 0  # start search from city 0
		path = [self.cities[cityIndex]]
		size = len(self.cities)
		table = np.empty([size, size])
		for i in range(size):
			for j in range(size):
				table[i, j] = (self.cities[i].costTo(self.cities[j]))
		initState = TSPState(cost, path, table, depth, cityIndex, np.array(self.cities))
		initState.reduce()
		return initState

	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

	def fancy(self, time_allowance=60.0):
		return self.greedy()



	def greedy(self, time_allowance=60.0):
		results = {}
		bssf = None
		foundTour = False
		count = 0
		start_time = time.time()
		for city in self.cities:
			newPath = self._findGreedyPath(city)
			if newPath is not None:
				if bssf is None or newPath.cost < bssf.cost:
					count += 1
					bssf = newPath
					foundTour = True
			if time.time() - start_time > time_allowance:
				break

		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results

	def _findGreedyPath(self, startCity):
		route = []
		visitedCities = []
		ncities = len(self.cities)
		for i in range(ncities):
			nextCity = self._getGreedyCity(startCity, visitedCities)
			if nextCity is None:
				return None
			route.append(nextCity)
			visitedCities.append(nextCity)
			startCity = nextCity
		return TSPSolution(route)

	def _getGreedyCity(self, startCity, visitedCities):
		min = math.inf
		bestCity = None
		for city in self.cities:
			if city not in visitedCities:
				length = startCity.costTo(city)
				if min > length:
					min = length
					bestCity = city
		return bestCity

	''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
