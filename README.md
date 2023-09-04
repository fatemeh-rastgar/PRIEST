Welcome to the **PRIEST** repository! This repository is associated with the paper **"PRIEST: Projection Guided Sampling-Based Optimization For Autonomous Navigation."** You can find the Arxiv pre-print of the paper here [..]. Additionally, we have provided videos and demos, which can be accessed [here](https://sites.google.com/view/priest-optimization)

**Contacts:**
- Fatemeh Rastgar (fatemeh.rastgar2@gmail.com)
- Arun Kumar Singh (aks1812@gmail.com)

**General Requirements:**

To run the codes, it is necessary to have: 

|ROS| Numpy|Jax-Numpy|(https://github.com/google/jax)|
|Ubuntu|Scipy|mayavi(https://docs.enthought.com/mayavi/mayavi/installation.html)|

Also, the following libraries and repositories have been used in our study

| Use| Repo/library name | Links |
| --- | --- |---|
| General | Numpy |...|
| General | Scipy |...|
| General | Jax-Numpy|(https://github.com/google/jax)|
|Plot 3d trajectories|mayavi|(https://docs.enthought.com/mayavi/mayavi/installation.html)|
|gradient-based solver | FATROP |(git@gitlab.kuleuven.be:robotgenskill/fatrop/fatrop.git)| 
|gradient-based solver |ROCKIT |(https://gitlab.kuleuven.be/meco-software/rockit)|
|Sampling-based method and Jackal| log-MPPI |(https://github.com/IhabMohamed/log-MPPI_ros)|

The first benchmark in our work is **Comparison on BARN Dataset**. To run our planner, we use the following commands: 

      roslaunch priest clearpath_launch.launch
      rosrun priest planner_holonomic.py

**RUN TEB/DWA planner**
* for Barn dataset:
  
        roslaunch comparison barn_world.launch
        roslaunch comparison map_less_navigation.launch
        rosrun comparison send_goal.py

Note that in mapl_less_navigation.launch file, we can choose if we use TEB planner or DWA planner.

**Run planner on Dynamic environment**

      roslaunch priest nonhol_clearpath.launch
      rosrun priest planner_nonhol_dy.py


 


We use the laser scan to observe the obstacles, so we update the following file in the jackal package.
 * The costmap_common_params.yaml
 * 
   

      
      

