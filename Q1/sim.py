import open3d as o3d
import numpy as np
import copy
import planner
from multiprocessing import Process, Manager, Value, Array, Lock
import math

# sampling time
DT = 0.01
# start point of bot in x, y, z
start_point = [1, 3, 1]
# destination point of the bot in x, y, z
destination_point = [20, 20, 10]
# number of random obstacles to spawn in the world
num_obstacles = 200
# bot dimensions in depth (x), width (y), height (z)
bot_size = [0.5, 0.6, 0.2]
# how far bot can sense obstacles
sensor_range = 10

# voxel size computed based on world size
voxel_size = math.pow(destination_point[0]*destination_point[1]*destination_point[2] / 4000, 0.3333)

# class to store bot's properties
class Bot:
    # intialise bot's properties
    def __init__(self, start_position, geometry, range):
        self.position = np.array(start_position, dtype=float)
        self.velocity = np.array([0, 0, 0], dtype=float)
        self.mesh_box = o3d.geometry.TriangleMesh.create_box(geometry[1], geometry[0], geometry[2])
        self.sensor_range = range

    # get bot's position
    def get_state(self):
        return self.position
    
    # give bot control velocity
    def set_control(self, v):
        self.velocity = v

    # simulate bot dynamics
    def update_state(self, dt):
        self.position += self.velocity*dt

    # get bot's dimension
    def get_geometry(self):
        self.mesh_box.compute_vertex_normals()
        self.mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
        self.mesh_box = self.mesh_box.translate((self.position[0], self.position[1], self.position[2]), relative=False)
        return self.mesh_box

# class to store world's properties
class World:
    # initiliase world properties
    def __init__(self, destination, num, dt):
        # list of obstacles in world
        self.obstacles = []
        # destination = opposite extreme of world
        self.destination = np.array(destination, dtype=float)
        self.dt = dt
        self.time = 0
        self.seen_obs = []
        self.unseen_obs = []
        self.seen_cloud = o3d.geometry.PointCloud()
        self.unseen_cloud = o3d.geometry.PointCloud()
        self.generate_random_obstacles(num)

    # add new obstacles to world
    def add_obstacle(self, obstacle):
        self.obstacles.append(np.array(obstacle, dtype = float))
        self.unseen_obs.append(len(self.obstacles)-1)

    # randomly generate obstacles
    def generate_random_obstacles(self, num):
        for i in range(num):
            self.add_obstacle(np.random.rand(3) * self.destination)

    # get obstacle in cloud format
    def get_obstacle_cloud(self):
        if len(self.unseen_obs)>0:
            self.unseen_cloud.points = o3d.utility.Vector3dVector(np.array([self.obstacles[i] for i in self.unseen_obs]))
            self.unseen_cloud.paint_uniform_color([0, 0, 0])
        if len(self.seen_obs)>0:
            self.seen_cloud.points = o3d.utility.Vector3dVector(np.array([self.obstacles[i] for i in self.seen_obs]))
            self.seen_cloud.paint_uniform_color([1, 0, 0])
        return self.seen_cloud, self.unseen_cloud

    # get the obstacles visible to the bot in shared memory
    def get_nearby_obstacles(self, bot, obs_list):
        obs_list[:] = []
        o_list = []
        for i, obs in enumerate(self.obstacles):
            if np.linalg.norm(obs - bot.get_state()) < bot.sensor_range:
                o_list.append(obs[0])
                o_list.append(obs[1])
                o_list.append(obs[2])
                if i not in self.seen_obs:
                    self.unseen_obs.remove(i)
                    self.seen_obs.append(i)
        obs_list[:] = o_list


    # get world sampling time
    def get_sampling_time(self):
        return self.dt

    # get world time
    def get_time(self):
        return self.time

    # update world time
    def update_time(self):
        self.time+=self.dt

    # world bounding box in lineset form for visualisation
    def get_bounding_box(self):
        points = [
            [0, 0, 0],
            [self.destination[0], 0, 0],
            [0, self.destination[1], 0],
            [self.destination[0], self.destination[1], 0],
            [0, 0, self.destination[2]],
            [self.destination[0], 0, self.destination[2]],
            [0, self.destination[1], self.destination[2]],
            [self.destination[0], self.destination[1], self.destination[2]],
        ]
        lines = [
            [0, 1],
            [0, 2],
            [1, 3],
            [2, 3],
            [4, 5],
            [4, 6],
            [5, 7],
            [6, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

# make path lineset for visualisation
def make_path_lineset (lineset, path):
    path_list = np.reshape(np.array(path), (int(len(path)/3), 3)).tolist()
    lines = []
    for i in range(len(path_list)-1):
        lines.append([i, i+1])
    colors = [[0, 0, 1] for i in range(len(lines))]
    lineset.points = o3d.utility.Vector3dVector(path_list)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector(colors)
    return lineset

# function to normalise a vector
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

# calculate and give control to bot based on current path
def give_control_input(vel, state, path):
    path_list = np.reshape(np.array(path), (int(len(path)/3), 3)).tolist()
    if len(path_list) > 0:
        dir_vector = np.array(path_list[0]) - np.array(state)
        vel[:] = (normalize(dir_vector)*2).tolist()
        if np.linalg.norm(np.array(state)-np.array(path_list[0])) < 0.5:
            path_list.pop(0)
            path[:] = np.ndarray.flatten(np.array(path_list)).tolist()
    else:
        vel[:] = np.zeros(3, dtype=float).tolist()

if __name__ == '__main__':
    # initialise world and bot
    world = World(destination_point, num_obstacles, DT)
    bot = Bot(start_point, bot_size, sensor_range)

    # setup visualiser
    path_line = o3d.geometry.LineSet()
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(bot.get_geometry())
    vis.add_geometry(world.get_bounding_box())
    vis.add_geometry(axes)
    seen, unseen = world.get_obstacle_cloud()
    vis.add_geometry(seen)
    vis.add_geometry(unseen)
    vis.add_geometry(path_line)
    ctr = vis.get_view_control()
    ctr.set_constant_z_far(10000)
    ctr.set_up(up=[-1, -1, -1])
    ctr.set_front(front=[1, 1, 1])

    # setup shared variables
    manager = Manager()
    lock = Lock()
    bot_control = Array('d', [0, 0, 0])
    obstacles_in_view = manager.list()
    path = manager.list()
    bot_state = Array('d', bot.get_state().tolist())
    voxel_size_shared = Value('d', voxel_size)
    bot_size_shared = Array('d', bot_size)
    world_bounds = Array('d', destination_point)

    # start astar path planning parallel process
    plan = Process(target=planner.get_path, args=(lock, bot_state, bot_size_shared, world_bounds, voxel_size_shared, obstacles_in_view, path))
    plan.start()

    # simulation loop
    while True:
        # get obstacles in bot's view
        world.get_nearby_obstacles(bot, obstacles_in_view)

        # calculate control input from path
        lock.acquire()
        give_control_input(bot_control, bot_state, path)
        lock.release()

        # give control input to bot
        bot.set_control(np.array(bot_control))
        # run bot dynamics
        bot.update_state(world.get_sampling_time())
        # update bot state shared variable
        bot_state[:] = bot.get_state().tolist()
        # update world time
        world.update_time()

        lock.acquire()

        # show world time and control input
        print ("time:", world.get_time(), "     control_input:", bot_control[:])

        # path visualisation
        make_path_lineset(path_line, path)
        lock.release()

        # update visualiser
        vis.update_geometry(path_line)
        seen, unseen = world.get_obstacle_cloud()
        vis.update_geometry(seen)
        vis.update_geometry(unseen)
        vis.update_geometry(bot.get_geometry())
        ctr = vis.get_view_control()
        vis.poll_events()
        vis.update_renderer()
    plan.join()