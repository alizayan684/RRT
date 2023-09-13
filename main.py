import csv
import random
from queue import PriorityQueue

import numpy


# output of RRT algorithm ##############################################################################################
def trace_path(goal_node):
    path = []
    current_node = goal_node
    while current_node.parent:
        path.append([round(current_node.x, 2), round(current_node.y, 2)])
        current_node = current_node.parent
    path.append([current_node.x, current_node.y])
    path.reverse()
    return path


# Read in obstacles from csv file and store in list ####################################################################
def read_obstacles():
    with open('BEST/obstacles.csv') as csvfile:
        reader = csv.reader(csvfile)
        obstacles = list(reader)
    csvfile.close()
    return obstacles


# make sample point function for RRT ###################################################################################
def sample_nodes():
    samples = PriorityQueue()  # initialize priority queue
    for samp_num in range(8):  # generate 8 random points
        x = random.uniform(-0.6, 0.6)
        y = random.uniform(-0.6, 0.6)
        point = [round(x, 2), round(y, 2)]
        samples.put(Node(point[0], point[1]))
    return samples


# make unit vector function for RRT ####################################################################################
def make_unit_vector(start_Node, end_Point):  # make unit vector for steering and obstacle checking
    if start_Node.x == end_Point[0] and start_Node.y == end_Point[1]:
        return [0, 0]
    vector = [end_Point[0] - start_Node.x, end_Point[1] - start_Node.y]
    vector = [(end_Point[0] - start_Node.x) / (numpy.sqrt(vector[0] ** 2 + vector[1] ** 2)),
              (end_Point[1] - start_Node.y) / (numpy.sqrt(vector[0] ** 2 + vector[1] ** 2))]

    return vector


# Node class ###########################################################################################################
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.children = []
        # calculate heuristic for priority queue (distance to goal) for sampling
        self.heuristic = numpy.sqrt((x - goal[0]) ** 2 + (y - goal[1]) ** 2)

    def __lt__(self, other):  # less than function for priority queue to compare sample points
        return self.heuristic < other.heuristic


# RRT class ############################################################################################################
class RRT:
    def __init__(self, start_point, goal_point, iterations, obstacles, max_length):
        self.START = Node(start_point[0], start_point[1])
        self.GOAL = Node(goal_point[0], goal_point[1])
        self.nearest_node = None
        self.iterations = iterations
        self.grid = obstacles
        self.step_size = max_length
        self.path = []
        self.nearest_distance = 10000

    def add_child(self, x, y):  # add child to nearest node
        if x == self.GOAL.x and y == self.GOAL.y:
            self.GOAL.parent = self.nearest_node
            self.nearest_node.children.append(self.GOAL)
        else:
            new_node = Node(x, y)
            new_node.parent = self.nearest_node
            self.nearest_node.children.append(new_node)

    def steer_towards(self, start_node, end_point):  # steer towards random point
        vector = make_unit_vector(start_node, end_point)
        vector = [vector[0] * self.step_size, vector[1] * self.step_size]
        new_Point = [start_node.x + vector[0], start_node.y + vector[1]]
        if new_Point[0] > 0.5:  # check if new point x is out of bounds
            new_Point[0] = 0.5

        if new_Point[0] < -0.5:
            new_Point[0] = -0.5

        if new_Point[1] > 0.5:  # check if new point y is out of bounds
            new_Point[1] = 0.5

        if new_Point[1] < -0.5:
            new_Point[1] = -0.5

        return new_Point

    def obstacle_found(self, nearest_Node, new_Point):  # check if obstacle is in path
        unit_vector = make_unit_vector(nearest_Node, new_Point)  # make unit vector
        test_point = [nearest_Node.x, nearest_Node.y]
        while self.distance(test_point, new_Point) > 0.05:  # check if obstacle is in path
            test_point[0] += unit_vector[0] * 0.05  # check every 0.05m
            test_point[1] += unit_vector[1] * 0.05  # check every 0.05m
            if self.is_point_in_obstacle(test_point):
                return True
        if self.is_point_in_obstacle(new_Point):
            return True
        return False

    def find_nearest_node(self, root, point):  # find nearest node to random point
        if not root:
            return
        root_point = [root.x, root.y]
        distance = self.distance(root_point, point)
        if distance < self.nearest_distance:
            self.nearest_node = root
            self.nearest_distance = distance
        if len(root.children):
            for child in root.children:
                self.find_nearest_node(child, point)
        return self.nearest_node

    def distance(self, point1, point2):  # calculate distance between two points
        return numpy.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def goal_reached(self, goal_node):  # check if goal is reached
        if self.nearest_node is not None:
            for child in self.nearest_node.children:
                if ((child.x - goal_node.x) ** 2 + (
                        child.y - goal_node.y) ** 2) <= 0.33:  # check if goal is reached within 0.37m
                    self.GOAL.parent = child
                    child.children.append(self.GOAL)
                    return True
        return False

    def reset_nearest_distance(self):  # reset nearest distance to a large number
        self.nearest_distance = 10000

    def is_point_in_obstacle(self, point):  # check if point is in obstacle
        for obstacle in self.grid:
            if ((point[0] - float(obstacle[0])) ** 2 + (point[1] - float(obstacle[1])) ** 2) <= float(
                    obstacle[2]) ** 2:  # check if point is in obstacle
                return True
        return False


if __name__ == '__main__':
    # inputs ##################################################
    path_points = []
    start = [-0.5, -0.5]
    goal = [0.5, 0.5]
    ITERATIONS = 1000000
    grid_obstacles = read_obstacles()
    step_size = 0.3
    ###########################################################
    rrt = RRT(start, goal, ITERATIONS, grid_obstacles, step_size)  # initialize RRT
    random_nodes = sample_nodes()  # initialize random nodes
    counter = 8  # counter for random nodes
    for i in range(ITERATIONS):
        counter -= 1
        if counter == 0:  # get new random nodes every 8 iterations
            counter = 8
            random_nodes = sample_nodes()
        rand_node = random_nodes.get()
        random_point = [rand_node.x, rand_node.y]
        nearest_node = rrt.find_nearest_node(rrt.START, random_point)
        new_point = rrt.steer_towards(nearest_node, random_point)
        if not rrt.obstacle_found(nearest_node, new_point):
            rrt.add_child(new_point[0], new_point[1])
            random_nodes = sample_nodes()
            rrt.reset_nearest_distance()
        if rrt.goal_reached(rrt.GOAL):
            path_points = trace_path(rrt.GOAL)
            break
        rrt.reset_nearest_distance()
    # end of RRT algorithm #########################################
    print(path_points)
    # write to nodes file ##########################################
    with open("BEST/nodes.csv", 'w') as n:
        writer = csv.writer(n)
        for i in range(len(path_points)):
            writer.writerow([i + 1, path_points[i][0], path_points[i][1]])
    # write to path file ##########################################
    li = []
    for i in range(len(path_points)):
        li.append(i + 1)
    with open("BEST/path.csv", 'w') as p:
        writer = csv.writer(p)
        writer.writerow(li)
    # write to edges file ##########################################
    with open("BEST/edges.csv", "w") as e:
        writer = csv.writer(e)
        for i in range(len(path_points) - 1):
            writer.writerow([i + 1, i + 2, 0.3])
    # END OF PROGRAM ###################################################################################################
