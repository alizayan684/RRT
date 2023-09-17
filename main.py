import csv
import math
import random

import numpy

# define needed constants ##############################################################################################
OBSTACLE_FILE_PATH = "BEST/obstacles.csv"  # the path to the obstacle definition csv file
NODES_FILE_PATH = "BEST/nodes.csv"  # the path to write a csv file of generated nodes
EDGES_FILE_PATH = "BEST/edges.csv"  # the path to write a csv file of generated edges
PATH_FILE_PATH = "BEST/path.csv"  # the path to write a csv file of the node path
GRAPH_MIN = -0.5  # minimum for generating a random sample point
GRAPH_MAX = 0.5  # maximum for generating a random sample point
MAX_LOOP_COUNT = 500  # defines the maximum number of attempts to find a path
SAMPLE_STEP_SIZE = 0.2  # defines the maximum step size to take when generating a new node
MIN_GOAL_DISTANCE = 0.001  # defines the error tolerance for checking if the goal has been reached
OBSTACLE_BUFFER = 0.02  # defines a small buffer distance to add to each obstacle for a better screenshot


########################################################################################################################
# Read in obstacles from csv file and store in list ####################################################################

def read_obstacles():
    with open(OBSTACLE_FILE_PATH) as csv_file:
        csv_reader = csv.reader(filter(lambda row_: row_[0] != '#', csv_file))
        obstacles = list(csv_reader)
    csv_file.close()
    return obstacles


# writing output in csv files ##########################################################################################
def write_nodes_csv(node_csv_file_path, nodes):
    lines = []
    for node in nodes:
        lines.append(f"{node.id}, {node.x:.4f}, {node.y:.4f}, 0\n")

    with open(node_csv_file_path, "w") as csv_file:
        csv_file.writelines(lines)


# write a list of edges to a csv file
def write_edges_csv(edges_csv_file_path, edges):
    lines = []
    for edge in edges:
        lines.append(f"{edge.node_id_1}, {edge.node_id_2}, {edge.cost}\n")

    with open(edges_csv_file_path, "w") as csv_file:
        csv_file.writelines(lines)


# walk up the node tree and write the path to the goal
def write_path_csv(path_csv_file, goal_node):
    result = []
    current_node = goal_node
    while current_node:
        result.append(current_node.id)
        current_node = current_node.parent

    result.reverse()
    num_results = len(result)
    with open(path_csv_file, "w") as csv_file:
        for i in range(0, num_results):
            csv_file.write(str(result[i]))
            if i + 1 < num_results:
                csv_file.write(",")


########################################################################################################################
def no_intersection(x1, y1, x2, y2, cx, cy, circle_radius):
    # Convert points to numpy arrays for easier math
    p1 = numpy.array([x1, y1])
    p2 = numpy.array([x2, y2])

    # Compute directional vector of line
    dp = p2 - p1
    magnitude = numpy.linalg.norm(dp)
    unit_vector = numpy.array([dp[0] / magnitude, dp[1] / magnitude])
    for i in range(1, 21):  # 20 points between p1 and p2
        new_point = p1 + unit_vector * (i / 20) * magnitude
        if ((new_point[0] - cx) ** 2 + (new_point[1] - cy) ** 2) ** 0.5 <= (
                circle_radius):  # if the distance between the point and the circle center is less than the radius
            return False

    return True


# Node class ###########################################################################################################
class Node:
    def __init__(self, x, y):
        self.id = 0
        self.x = x
        self.y = y
        self.parent = None
        self.children = []
        self.heuristic = ((GRAPH_MAX - x) ** 2 + (GRAPH_MAX - y) ** 2) ** 0.5  # euclidean distance to goal


# defines a class representing an edge between two nodes of the graph
class Edge:
    def __init__(self, node_id_1, node_id_2):
        self.node_id_1 = node_id_1
        self.node_id_2 = node_id_2
        self.cost = 0


# RRT class ############################################################################################################
class RRT:
    def __init__(self):
        self.node_count = 1  # the number of nodes in the tree used to assign unique node ids
        self.nodes = []
        self.edges = []

    # the tree is responsible for ensuring that each added node has a unique ID
    def add_node(self, node):
        node.id = self.node_count
        self.nodes.append(node)
        self.node_count = self.node_count + 1  # increment the node count used for the next node id
        return node.id

    def add_edge(self, edge):
        self.edges.append(edge)


# end of class #########################################################################################################
# functions ############################################################################################################
random_points = []
for i in range(-50, 51):  # generate a list of 10,000 random points to sample from representing the graph
    for j in range(-50, 51):
        random_points.append([i / 100, j / 100])


def get_sample():
    point = random.choice(random_points)
    return point


def check_motion_is_collision_free(obstacle_list, start_x, start_y, end_x, end_y):
    for obstacle in obstacle_list:
        if no_intersection(start_x, start_y, end_x, end_y, float(obstacle[0]), float(obstacle[1]),
                           float(obstacle[2]) / 2.0 + OBSTACLE_BUFFER):
            continue
        else:
            return False
    return True


def get_closest_tree_node(tree, x, y):
    min_distance = math.inf
    closest_node = None
    for node in tree.nodes:
        distance = ((x - node.x) ** 2 + (y - node.y) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_node = node
    return closest_node


def local_planner(start_x, start_y, target_x, target_y, step_size):
    # find the normalized vector from the start point to the target
    start = numpy.array([start_x, start_y])
    target = numpy.array([target_x, target_y])
    direction = target - start

    magnitude = numpy.linalg.norm(direction)
    if magnitude > step_size:  # if the distance between the start and target is greater than the step size
        direction = direction / magnitude
        direction = direction * step_size  # scale the direction vector to the step size

    # return the adjusted target point
    new = start + direction
    return new[0], new[1]


########################################################################################################################
# main program #########################################################################################################
if __name__ == '__main__':
    # read in the obstacles
    obstacles = read_obstacles()

    # create the RRT tree
    rrt = RRT()

    # create the start and goal nodes
    start_node = Node(-0.5, -0.5)
    goal_node = Node(0.5, 0.5)

    # add the start node to the tree
    rrt.add_node(start_node)

    # loop until the goal is reached or the maximum number of iterations is reached
    goal_reached = False
    loop_count = 0
    while not goal_reached and loop_count < MAX_LOOP_COUNT:
        print(loop_count)
        # get a random sample point
        if loop_count % 10 == 0:  # chose the goal as the sample point every 10 iterations to bias the search
            sample_point = [0.5, 0.5]

        else:
            sample_point = get_sample()

        # find the closest node in the tree to the sample point
        closest_node = get_closest_tree_node(rrt, sample_point[0], sample_point[1])

        # find the new point to add to the tree
        new_point = local_planner(closest_node.x, closest_node.y, sample_point[0], sample_point[1], SAMPLE_STEP_SIZE)

        # check if the new point is collision free
        if closest_node.x == new_point[0] and closest_node.y == new_point[1]:
            continue
        if check_motion_is_collision_free(obstacles, closest_node.x, closest_node.y, new_point[0], new_point[1]):
            # create a new node and add it to the tree
            new_node = Node(new_point[0], new_point[1])
            new_node.parent = closest_node
            closest_node.children.append(new_node)
            rrt.add_node(new_node)

            # create a new edge and add it to the tree
            new_edge = Edge(closest_node.id, new_node.id)
            rrt.add_edge(new_edge)
            # check if the goal has been reached
            if ((new_node.x - goal_node.x) ** 2 + (new_node.y - goal_node.y) ** 2) ** .5 < MIN_GOAL_DISTANCE:
                goal_reached = True
                # write the nodes and edges to csv files
                write_nodes_csv(NODES_FILE_PATH, rrt.nodes)
                write_edges_csv(EDGES_FILE_PATH, rrt.edges)
                write_path_csv(PATH_FILE_PATH, new_node)
                print("Goal Reached")

        # increment the loop counter
        loop_count = loop_count + 1

# END OF PROGRAM ###################################################################################################
