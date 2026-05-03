# Doulbe Jointed Arm Configuration Space Motion Planner
Motion planning — finding a collision-free path for a robot arm — is a fundamental problem in robotics. This project implements it from scratch using configuration space theory and the RRT algorithm, with an interactive visualization.

This project creates and graphs a 2d configuration space for a double-jointed arm with circular and polygonal obstacles. 
The configuration space is a 2d coordinate plane with x and y axis representing the angles of the two segments of the arm. 
Each point in the space represents a unique position of the arm.
This project can generate paths between two points in the configuration space that avoids circular obstacles. 

## arm.py
This file contains calculates elbow and tip coordinates of the arm using basic trigonometry. 
It also has methods to check whether the arm intersects circular obstacles using vector projections.


## pathing.py
This file uses the rapidly exploring random tree algorithm to generate a path between a start position and end position on the configurations space. 
Essentially, this algorithm creates a tree of nodes and every step, 
it chooses a random direction and moves some small distance in that direction from the nearest node in the tree. 
If the step does not pass through any obstacles, then it adds it to the tree. 
It keeps doing this until getting within a step distance to the ending point.
It contains a method to smooth the path using a path shortcutting algorithm that greedily removes unnecessary waypoints by checking if consecutive nodes can be connected directly.

## animation.py
This file graphs the configuration space using matplotlib by checking if each pixel is inside an obstacle. 
Users can select two starting points in the configuration space and generate a rrt path between them.
The project will animate the arm as the path moves forward in time.

## Math
- Forward kinematics: derived using trigonometry and vector addition
- Collision detection: segment-circle intersection via vector dot product projection
- C-space sampling: toroidal distance metric to handle angle wraparound
- RRT: probabilistically complete, finds path with probability 1 if one exists

## Usage
pip install numpy matplotlib
run animation.py to simulate the arm 
