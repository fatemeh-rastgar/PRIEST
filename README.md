# PRIEST
Projection Guided Sampling-Based Optimization For Autonomous Navigation 

Repository associated with the paper **"PRIEST: Projection Guided Sampling-Based Optimization For Autonomous Navigation."** Arxiv pre-print can be found here (link).

Contacts: Fatemeh Rastgar (fatemeh.rastgar2@gmail.com), Arun Kumar Singh (aks1812@gmail.com)

**General Requirements:**

|ROS installation| Numpy|Jax-Numpy|(https://github.com/google/jax)|
|Ubuntu|Scipy|mayavi(https://docs.enthought.com/mayavi/mayavi/installation.html)|

plus the following libraries and repositories 

| Use| Repo/library name | Links |
| --- | --- |---|
| General | Numpy |...|
| General | Scipy |...|
| General | Jax-Numpy|(https://github.com/google/jax)|
|Plot 3d trajectories|mayavi|(https://docs.enthought.com/mayavi/mayavi/installation.html)|
|gradient-based solver | FATROP |(git@gitlab.kuleuven.be:robotgenskill/fatrop/fatrop.git)| 
|gradient-based solver |ROCKIT |(https://gitlab.kuleuven.be/meco-software/rockit)|
|Sampling-based method| log-MPPI |(https://github.com/IhabMohamed/log-MPPI_ros)|

** Run planner on Barn dataset ***

      roslaunch priest clearpath_launch.launch
      
      rosrun priest planner_holonomic.py
      

