import numpy as np
import heapq

# convert from world position to astar grid index
def index_to_world(index, voxel_size):
    return np.array(index) * voxel_size

# convert from grid index to world position 
def world_to_index(pos, voxel_size):
    ind = (np.array(pos) / voxel_size).astype(int)
    return (ind[0], ind[1], ind[2]) #tuple((np.array(pos) / voxel_size).astype(int).tolist())

# function to fill the costmap with obstacles
def fill_costmap(costmap, obs, bot_size, bound, voxel_size):
    voxel_bot_size = ((np.array(bot_size) + 2*voxel_size) / voxel_size).astype(int)
    # fil the obstacle's voxel and nearby voxels according to bot's dimensions
    for i in range(-voxel_bot_size[0] // 2, voxel_bot_size[0 // 2]):
        for j in range(-voxel_bot_size[1] // 2, voxel_bot_size[1] // 2):
            for k in range(-voxel_bot_size[2] // 2, voxel_bot_size[2] // 2):
                obs_index = world_to_index(obs, voxel_size)
                if in_bounds(obs_index, bound):
                    costmap[obs_index[0] + i, obs_index[1] + j, obs_index[2] + k] = 1

# returns the path to traverse on using astar
def get_path(lock, state, bot_size, bounds, voxel_size, obstacles, path_list):
    lock.acquire()
    # create costmap
    cost_map = np.zeros( (np.array(bounds) // voxel_size.value + np.array([1, 1, 1])).astype(int) )
    voxel = voxel_size.value
    destination = bounds[:]
    lock.release()
    while True:
        lock.acquire()
        obstacle_list = obstacles[:]
        lock.release()
        
        # get obstacles from shared variables
        if len(obstacle_list) > 0:
            while len(obstacle_list) % 3 != 0:
                obstacle_list.pop()
        len_obs = len(obstacle_list)
        obstacle_list = np.array(obstacle_list)
        obstacle_list = np.reshape(obstacle_list, (int(len_obs / 3), 3))
        obstacle_list = np.atleast_2d(obstacle_list)
    
        # fill costmap with obstacles
        if len_obs > 0:
            lock.acquire()
            for obs in obstacle_list:
                fill_costmap(cost_map, obs, bot_size[:], bounds[:], voxel)
            lock.release()

        # get start node from position or path
        lock.acquire()
        start_node = world_to_index(state[:], voxel)
        if len(path_list) > 6 and np.linalg.norm(np.array(state[:]) - np.array(destination)) > 10:
            path_formatted = np.reshape(np.array(path_list), (int(len(path_list)/3), 3)).tolist()
            start_node = world_to_index(path_formatted[1], voxel)
        lock.release()
        
        target_node = world_to_index(np.array(destination)-1, voxel)
        
        lock.acquire()
        pos = state[:]
        lock.release()
        
        # find path using astart and change the shared variable
        if np.linalg.norm(np.array(pos) - np.array(destination)) > voxel:
            path = astar(start_node, target_node, cost_map)
            path_scaled = [index_to_world(index, voxel) for index in path]
            lock.acquire()
            path_list[:] = np.ndarray.flatten(np.array(path_scaled)).tolist()
            lock.release()
        else:
            print ("REACHED")

# get neighbours indices of a grid cell
def get_neighbours (index):
    neighbour_list = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i != 0 or j != 0 or k != 0:
                    neighbour_list.append((index[0] + i, index[1] + j, index[2] + k))
    return neighbour_list

# add node to open list in astar
def add_to_open(open_nodes, nodes, add_node):
    open_nodes.append(add_node)
    nodes[add_node][2] = 1

# add node to closed list and remove form open list in astar
def add_to_closed(open_nodes, closed_nodes, nodes, add_node):
    open_nodes.remove(add_node)
    closed_nodes.append(add_node)
    nodes[add_node][2] = 2

# get node with lowest cost
def get_optimal_node(open_nodes, nodes):
    min_cost = nodes[open_nodes[0]][0]
    min_index = open_nodes[0]
    for i in open_nodes:
        if nodes[i][1] < min_cost:
            min_cost = nodes[i][0]
            min_index = i
    return min_index

# check if there is a shorter path available to neighbour node
def has_shorter_path(cur, neighbour, nodes):
    if nodes[neighbour][2] != 1:
        return False
    elif nodes[cur][0] + np.linalg.norm(np.array(neighbour) - np.array(cur)) < nodes[neighbour][0]:
        return True
    else:
        return False

# get path length to given node
def get_gcost(cur, neighbour, nodes):
    return nodes[cur][0] + np.linalg.norm(np.array(neighbour) - np.array(cur))

# get cost of given node
def get_fcost(cur, target, nodes):
    return nodes[cur][0] + np.linalg.norm(np.array(target) - np.array(cur))

# check if given index is inside grid
def in_bounds(nod, bound):
    if  nod[0] >= 0 and nod[1] >= 0 and nod[2] >= 0 and nod[0] <= bound[0] and nod[1] <= bound[1] and nod[2] <= bound[2]:
        return True
    else:
        return False

# implementation of astar algorithm
def astar(start_node, target_node, cost_map):
    nodes = np.zeros((cost_map.shape[0], cost_map.shape[1], cost_map.shape[2], 3), dtype=float)
    parents = np.ones((cost_map.shape[0], cost_map.shape[1], cost_map.shape[2], 3), dtype=int)
    parents = parents * (-1)
    open_nodes = []
    closed_nodes = []

    add_to_open(open_nodes, nodes, start_node)
    nodes[start_node][0] = 0
    nodes[start_node][1] = np.linalg.norm(np.array(target_node) - np.array(start_node))
    parents[start_node] = start_node

    # loop until path is found
    while True:
        if len(open_nodes) > 0:
            current_node = get_optimal_node(open_nodes, nodes)
            add_to_closed(open_nodes, closed_nodes, nodes, current_node)
        else:
            print ("REACHED")

        if current_node == target_node:
            break
        
        for neighbour in get_neighbours(current_node):
            if in_bounds(neighbour, target_node) and cost_map[neighbour] == 0 and nodes[neighbour][2] != 2:
                if has_shorter_path(current_node, neighbour, nodes) or nodes[neighbour][2] != 1:
                    nodes[neighbour][0] = get_gcost(current_node, neighbour, nodes)
                    nodes[neighbour][1] = get_fcost(neighbour, target_node, nodes)
                    parents[neighbour] = current_node
                    if nodes[neighbour][2] != 1:
                        add_to_open(open_nodes, nodes, neighbour)

    # retrace the parents to find path
    path = []
    cur = target_node
    while tuple(parents[cur]) != cur:
        path.append(cur)
        cur = tuple(parents[cur])

    path.reverse()
    return path