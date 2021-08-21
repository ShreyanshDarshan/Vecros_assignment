# Vecros_assignment

## Dependencies

I have used Intel ISL's Open3D library for pointcloud read/write and robot path planning visualisation.
To install Open3D, run:
```
pip install open3d
```
This should install smoothly, but if there are any errors refer [here](http://www.open3d.org/docs/release/getting_started.html) or [here](https://pypi.org/project/open3d/)


Install numpy if you dont have it installed already:
```
pip install numpy
```

## Installation
```
git clone https://github.com/ShreyanshDarshan/Vecros_assignment.git
```

## How to run
To run question 1, run the commands:
```
cd Vecros_assignment/Q1
python sim.py
```

To run question 2, run the commmands:
```
cd Vecros_assignment/Q2
python segment_plane.py
```

## Outputs
### For q1:
Black obstacles are outside drone's field of view

Red obstacles are the ones drone has seen already
![q1 output](https://github.com/ShreyanshDarshan/Vecros_assignment/blob/main/q1_output.gif)


### For q2:
Green Plane is the Largest Plane

Red points are the points not lying on the largest plane

Surface area gets printed on the terminal when you run the program
![q2 output](https://github.com/ShreyanshDarshan/Vecros_assignment/blob/main/q2_output.png)
