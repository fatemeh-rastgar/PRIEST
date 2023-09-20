Welcome to the **PRIEST** repository! This repository is associated with the paper **"PRIEST: Projection Guided Sampling-Based Optimization For Autonomous Navigation."**. Additionally, we have provided videos and demos, which can be accessed [here](https://sites.google.com/view/priest-optimization). 

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
|Holonomic robot |(robootont_description, robotont_gazebo)| (https://github.com/robotont)|

**Running the First Benchmark - Comparison on BARN Dataset**

For the first benchmark, we conducted comparisons between several planners, including DWA, TEB, MPPI, log-MPPI, and our PRIEST method.
To run these comparisons, follow these steps:

1- Launch PRIEST

      roslaunch priest clearpath_launch.launch
      rosrun priest planner_holonomic.py

2- Launch DWA/TEB Planner
  
        roslaunch compare barn_world.launch
        roslaunch compare map_less_navigation.launch
        rosrun compare send_goal.py

Note that in the comparison/launch/map_less_navigation.launch file, you can choose whether to use the TEB planner or the DWA planner. To use the planner, use:

      <include file="$(find compare)/launch/move_base_teb.launch">
      
and to use the DWA planner:

      <include file="$(find compare)/launch/move_base.launch">
      
3- Launch MPPI/ log-MPPI:
      roslaunch priest clearpath_launch.launch
      roslaunch compare control_stage_robotont.launch
      rosrun compare send_goal.py

Please change the value from "true" to "false" in the following command within the control_stage_robotont.launch file to select the log-MPPI or MPPI method: 

       <arg name="normal_dist" default="false" />

**Running the Planner on a Dynamic Environment**

For this benchmark, we conducted comparisons between several planners, including baseline CEM, DWA, TEB, MPPI, log-MPPI, and our PRIEST method.
To run these comparisons, follow these steps:

1- Launch PRIEST:

      roslaunch priest nonhol_clearpath.launch
      rosrun priest planner_nonhol_dy.py

2- Launch CEM:

      roslaunch priest nonhol_clearpath.launch
      rosrun priest planner_cem_dynamic.py

3- Launch TEB/DWA Planner

      will be addded

4- Launch MPPI/ log-MPPI

      roslaunch compare clear_path_dynamic.launch 
      roslaunch mppi_control control_stage.launch
      rosrun compare send_obs_velocities.py
      rosrun compare send_goal.py

Please note that the following changes are required in send_goal.py 

- Subscriber topic: "/odom" to "/ground_truth/odom"
  
- goal.target_pose.header.frame_id should be "/map"



 We use laser scans to observe obstacles, so make sure to update the following files in the Jackal package:

- costmap_common_params.yaml

**Point-to-point-navigation**
### 2D Comparison
To perform a 2D comparison, follow these steps:

1. Navigate to the `2d_comparison` folder:
2. Run the `main.py` script:

### 3D Comparison
To perform a 2D comparison, follow these steps:

1. Navigate to the `3d_comparison` folder:
2. Run the `main.py` script:

  
If you have any questions or need further assistance, please feel free to contact us.
      
      

