import numpy as np

# Function to calculate relative transformation between two poses (already defined)
def pose_to_pose_relative(pose_A, pose_B):
    x_A, y_A, yaw_A = pose_A
    x_B, y_B, yaw_B = pose_B
    
    # Step 1: Compute global displacement
    delta_x_G = x_B - x_A
    delta_y_G = y_B - y_A
    
    # Step 2: Apply inverse rotation of yaw_A to get the displacement in the local frame of A
    cos_yaw_A = np.cos(-yaw_A)
    sin_yaw_A = np.sin(-yaw_A)
    
    delta_x_L = delta_x_G * cos_yaw_A - delta_y_G * sin_yaw_A
    delta_y_L = delta_x_G * sin_yaw_A + delta_y_G * cos_yaw_A
    
    # Step 3: Compute relative yaw and normalize it to [-pi, pi]
    delta_yaw = yaw_B - yaw_A
    delta_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))  # Normalize to [-pi, pi]
    
    return np.array([delta_x_L, delta_y_L, delta_yaw])

def pose_to_pose_relative_v2(pose_A, pose_B):
    # Convert poses to homogeneous transformation matrices
    x_A, y_A, yaw_A = pose_A
    x_B, y_B, yaw_B = pose_B
    
    # Create transformation matrix for pose A
    T_A = np.array([
        [np.cos(yaw_A), -np.sin(yaw_A), x_A],
        [np.sin(yaw_A), np.cos(yaw_A), y_A],
        [0, 0, 1]
    ])
    
    # Create transformation matrix for pose B
    T_B = np.array([
        [np.cos(yaw_B), -np.sin(yaw_B), x_B],
        [np.sin(yaw_B), np.cos(yaw_B), y_B],
        [0, 0, 1]
    ])
    
    # Calculate relative transformation matrix
    T_rel = np.linalg.inv(T_A) @ T_B
    
    # Extract relative x, y, and yaw from transformation matrix
    x_rel = T_rel[0,2]
    y_rel = T_rel[1,2]
    yaw_rel = np.arctan2(T_rel[1,0], T_rel[0,0])
    
    return np.array([x_rel, y_rel, yaw_rel])

# Function to integrate relative transformations back to absolute poses (already defined)
def integrate_relative_poses(initial_pose, relative_poses):
    poses = [initial_pose]
    current_pose = np.copy(initial_pose)
    
    def pose_to_matrix(x, y, theta):
        """Convert pose to 3x3 transformation matrix"""
        return np.array([
            [np.cos(theta), -np.sin(theta), x],
            [np.sin(theta), np.cos(theta), y],
            [0, 0, 1]
        ])
    
    def matrix_to_pose(matrix):
        """Convert 3x3 transformation matrix to pose"""
        x = matrix[0,2]
        y = matrix[1,2]
        theta = np.arctan2(matrix[1,0], matrix[0,0])
        return np.array([x, y, theta])

    # Convert initial pose to matrix
    current_matrix = pose_to_matrix(current_pose[0], current_pose[1], current_pose[2])
    
    for relative_pose in relative_poses:
        # Convert relative pose to matrix
        dx, dy, dyaw = relative_pose
        relative_matrix = pose_to_matrix(dx, dy, dyaw)
        
        # Multiply matrices to get new absolute pose
        current_matrix = current_matrix @ relative_matrix
        
        # Convert back to pose representation
        current_pose = matrix_to_pose(current_matrix)
        poses.append(current_pose)
    
    return np.array(poses)

# Function to calculate relative poses for entire trajectory (already defined)
def calculate_relative_trajectory(trajectory):
    poses = trajectory.poses
    num_poses = poses.shape[0]
    
    relative_poses = []
    
    for i in range(1, num_poses):
        relative_pose = pose_to_pose_relative_v2(poses[i-1], poses[i])
        relative_poses.append(relative_pose)
    
    return np.array(relative_poses)

def calculate_relative_pose(poses):
    num_poses = poses.shape[0]
    relative_poses = []
    for i in range(1, num_poses):
        relative_pose = pose_to_pose_relative(poses[i-1], poses[i])
        relative_poses.append(relative_pose)
    return np.array(relative_poses)

# Function to concatenate history and future trajectories
def concatenate_trajectories(history_trajectory, future_trajectory):
    # Concatenate the poses of history and future trajectories
    concatenated_poses = np.vstack((history_trajectory.poses, future_trajectory.poses))
    return Trajectory(concatenated_poses)

# Test function to check if integration of relative poses returns the original absolute poses (already defined)
def test_integration_of_relative_poses(trajectory):
    # Get the absolute poses
    absolute_poses = trajectory.poses
    
    # Calculate the relative poses
    relative_poses = calculate_relative_trajectory(trajectory)
    print("relative_poses\n", relative_poses)
    
    # Integrate the relative poses starting from the initial pose
    initial_pose = absolute_poses[0]
    integrated_poses = integrate_relative_poses(initial_pose, relative_poses)
    
    # Compare the integrated poses with the original absolute poses
    assert np.allclose(integrated_poses, absolute_poses, atol=1e-6), "Integration test failed!"
    print("Integration test passed!")

# History trajectory (already defined)
history_trajectory = np.array([[-8.46101595,  2.51522769, -0.57974404],
                               [-5.85742765,  1.14676025, -0.36981378],
                               [-3.15109776,  0.34361281, -0.17540229],
                               [ 0.        ,  0.        ,  0.        ]])
future_trajectory = np.array([[ 3.31196956,  0.19236557,  0.12221887],
                              [ 6.78643427,  0.6841803 ,  0.15372734],
                              [10.39714139,  1.25637799,  0.16436028],
                              [14.24632329,  1.8701929 ,  0.16641822],
                              [18.28015932,  2.54556115,  0.170002  ],
                              [22.52583992,  3.29488256,  0.16787723],
                              [26.69792036,  3.99717862,  0.16353954],
                              [31.07267531,  4.71792568,  0.15949101],
                              [35.35556618,  5.42577572,  0.15836009],
                              [39.52084817,  6.11954557,  0.15693663]])

# Define the Trajectory class (already defined)
class Trajectory:
    def __init__(self, poses):
        self.poses = poses

# Create trajectory objects for history and future
history_traj = Trajectory(history_trajectory)
future_traj = Trajectory(future_trajectory)

# Concatenate history and future trajectories
concatenated_traj = concatenate_trajectories(history_traj, future_traj)

# Run the test on the concatenated trajectory
print("Testing concatenated trajectory integration:")
test_integration_of_relative_poses(concatenated_traj)