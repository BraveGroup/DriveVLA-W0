from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, Scene, SensorConfig
import glob
import os
import numpy as np
import json
import argparse
from typing import Optional

class EmuVLAAgent(AbstractAgent):
    """External agent interface."""

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        experiment_path: str = '/default/path/to/experiment',
        anchor_based: bool = False,
    ):
        """
        Initializes the external agent object.
        :param trajectory_sampling: trajectory sampling specification
        :param experiment_path: path to the experiment results
        :param anchor_based: whether to use anchor-based trajectory (relative action)
        """
        self._trajectory_sampling = trajectory_sampling
        self.experiment_path = experiment_path
        self.anchor_based = anchor_based

    def name(self) -> str:
        """Inherited, see superclass."""

        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_no_sensors()
        # return SensorConfig.build_all_sensors(include=[0,1,2,3])

    def regular_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def convert_relative_to_absolute_se2_trajectory(
        self,
        rel_traj
    ):
        """
        Convert relative SE2 trajectory to absolute trajectory.
        Assumes initial pose is (0, 0, 0).
        
        :param rel_traj: [N, 3] array of (dx, dy, dtheta) in local frame of previous timestep
        :return: [N, 3] array of absolute (x, y, yaw)
        """
        N = rel_traj.shape[0]
        abs_traj = np.zeros_like(rel_traj)

        x, y, yaw = 0.0, 0.0, 0.0

        for i in range(N):
            dx, dy, dyaw = rel_traj[i]

            # Rotate local motion to global frame
            global_dx = np.cos(yaw) * dx - np.sin(yaw) * dy
            global_dy = np.sin(yaw) * dx + np.cos(yaw) * dy

            # Update global pose
            x += global_dx
            y += global_dy
            yaw = self.regular_angle(yaw + dyaw)

            abs_traj[i] = [x, y, yaw]

        return abs_traj

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene, action_type: Optional[str]=None) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param agent_input: Dataclass with agent inputs.
        :param scene: Current scene information.
        :return: Trajectory representing the predicted ego's position in future
        """
        experiment_path = self.experiment_path
        # load action here
        self.cur_idx = 3
        token = scene.frames[self.cur_idx].token
        action_file_path = os.path.join(experiment_path, f"{token}.json")
        rel_action = json.load(open(action_file_path, "r"))["action"]  # [8,3] relative action
        # rel_action = json.load(open(action_file_path, "r"))["action_gt_denorm_decode"]  # [8,3] relative action

        # Convert relative action to absolute trajectory
        rel_action = np.array(rel_action, dtype=np.float64)  # [8,3]
        if action_type is None:
            abs_action = self.convert_relative_to_absolute_se2_trajectory(rel_action)  # [8,3]
        elif action_type == "absolute":
            # 深拷贝 rel_action，确保 abs_action 是独立的副本
            abs_action = rel_action.copy()
            # 使用矢量化操作归一化 yaw
            abs_action[:, 2] = np.vectorize(self.regular_angle)(abs_action[:, 2])
        else:
            raise ValueError(f"Unknown action type: {action_type}. Expected values are None or 'absolute'.")
        
        # 根据 anchor_based 开关选择使用相对动作还是绝对动作
        if self.anchor_based:
            agent_trajectory = Trajectory(rel_action)
        else:
            agent_trajectory = Trajectory(abs_action)

        ################ For visualization ################
        human_trajectory = scene.get_future_trajectory(self._trajectory_sampling.num_poses)

        # Compute differences
        diff = agent_trajectory.poses - human_trajectory.poses
        # Calculate per-timestep absolute differences
        diff_x = np.abs(diff[:, 0]).tolist()
        diff_y = np.abs(diff[:, 1]).tolist()
        diff_yaw = np.abs(diff[:, 2]).tolist()

        diff_data = {
            "diff_x": diff_x,
            "diff_y": diff_y,
            "diff_yaw": diff_yaw
        }
        
        # Write differences to the figure directory
        scene_token = scene.scene_metadata.initial_token
        diff_file_path = os.path.join(experiment_path, f"{scene_token}_agent_human_traj_diff.json")
        with open(diff_file_path, 'w') as f:
            json.dump(diff_data, f, indent=4)

        from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
        from navsim.visualization.plots import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, configure_bev_ax, configure_ax
        import matplotlib.pyplot as plt

        frame_idx = scene.scene_metadata.num_history_frames - 1  # current frame
        fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
        add_configured_bev_on_ax(ax, scene.map_api, scene.frames[frame_idx])
        add_trajectory_to_bev_ax(ax, human_trajectory, TRAJECTORY_CONFIG["human"])
        add_trajectory_to_bev_ax(ax, agent_trajectory, TRAJECTORY_CONFIG["agent"])
        configure_bev_ax(ax)
        configure_ax(ax)
        # save fig
        fig.savefig(f'{experiment_path}/{scene_token}.png')
        plt.clf()

        return agent_trajectory

def calculate_average_differences(experiment_path: str):
    """
    Calculates and prints the average x, y, yaw differences from all diff files in the experiment_path.
    """
    diff_files = glob.glob(os.path.join(experiment_path, "*_diff.json"))
    total_diff_x = 0.0
    total_diff_y = 0.0
    total_diff_yaw = 0.0
    count = 0

    diff_x_list = []
    diff_y_list = []
    diff_yaw_list = []
    
    for diff_file in diff_files:
        with open(diff_file, 'r') as f:
            diff_data = json.load(f)
            total_diff_x += abs(diff_data["diff_x"])
            total_diff_y += abs(diff_data["diff_y"]) 
            total_diff_yaw += abs(diff_data["diff_yaw"])
            diff_x_list.append(abs(diff_data["diff_x"]))
            diff_y_list.append(abs(diff_data["diff_y"]))
            diff_yaw_list.append(abs(diff_data["diff_yaw"]))
            count += 1
            
    # Plot distributions
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.hist(diff_x_list, bins=128)
    ax1.set_title('X Difference Distribution')
    ax1.set_xlabel('Absolute X Difference')
    ax1.set_ylabel('Count')
    
    ax2.hist(diff_y_list, bins=128)
    ax2.set_title('Y Difference Distribution') 
    ax2.set_xlabel('Absolute Y Difference')
    ax2.set_ylabel('Count')
    
    ax3.hist(diff_yaw_list, bins=128)
    ax3.set_title('Yaw Difference Distribution')
    ax3.set_xlabel('Absolute Yaw Difference')
    ax3.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(experiment_path, 'agent_human_traj_diff_distribution.png'))
    plt.close()

    if count == 0:
        print("No difference files found.")
        return

    average_diff_x = total_diff_x / count
    average_diff_y = total_diff_y / count
    average_diff_yaw = total_diff_yaw / count

    print(f"Average Differences - x: {average_diff_x:.4f}, y: {average_diff_y:.4f}, yaw: {average_diff_yaw:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Calculate average differences from trajectory diff files.")
    parser.add_argument('experiment_path', type=str, help='Path to the experiment directory containing diff files.')
    args = parser.parse_args()

    calculate_average_differences(args.experiment_path)

if __name__ == "__main__":
    main()
