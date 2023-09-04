Welcome to the **PRIEST** repository! This repository is associated with the paper **"PRIEST: Projection Guided Sampling-Based Optimization For Autonomous Navigation."** You can find the Arxiv pre-print of the paper here [..]. Additionally, we have provided videos and demos, which can be accessed [here](https://sites.google.com/view/priest-optimization)

**Contacts:**
- Fatemeh Rastgar (fatemeh.rastgar2@gmail.com)
- Arun Kumar Singh (aks1812@gmail.com)

**General Requirements:**

To run the codes, you'll need the following dependencies:
- ROS
- Numpy
- Jax-Numpy (https://github.com/google/jax)
- Ubuntu
- Scipy
- mayavi(https://docs.enthought.com/mayavi/mayavi/installation.html)

We have also utilized the following libraries and repositories in our study:

| Use| Repo/library name | Links |
| --- | --- |---|
| General | Numpy |...|
| General | Scipy |...|
| General | Jax-Numpy|(https://github.com/google/jax)|
|Plot 3d trajectories|mayavi|(https://docs.enthought.com/mayavi/mayavi/installation.html)|
|gradient-based solver | FATROP |(git@gitlab.kuleuven.be:robotgenskill/fatrop/fatrop.git)| 
|gradient-based solver |ROCKIT |(https://gitlab.kuleuven.be/meco-software/rockit)|
|Sampling-based method and Jackal| log-MPPI |(https://github.com/IhabMohamed/log-MPPI_ros)|

**Running the First Benchmark - Comparison on BARN Dataset**

For the first benchmark, we conducted comparisons between several planners, including DWA, TEB, MPPI, log-MPPI, and our PRIEST method.
To run these comparisons, follow these steps:

1- Launch PRIEST

      roslaunch priest clearpath_launch.launch
      rosrun priest planner_holonomic.py

2- Launch DWA/TEB Planner
  
        roslaunch comparison barn_world.launch
        roslaunch comparison map_less_navigation.launch
        rosrun comparison send_goal.py

Note that in the comparison/launch/map_less_navigation.launch file, you can choose whether to use the TEB planner or the DWA planner. To use the planner, use:

      <include file="$(find robotont_nav)/launch/move_base_teb.launch">
      
and to use the DWA planner:

      <include file="$(find robotont_nav)/launch/move_base.launch">
      


**Running the Planner on a Dynamic Environment**

      roslaunch priest nonhol_clearpath.launch
      rosrun priest planner_nonhol_dy.py


 


We use the laser scan to observe the obstacles, so we update the following file in the jackal package.
 * The costmap_common_params.yaml
 * 
   

      
      

