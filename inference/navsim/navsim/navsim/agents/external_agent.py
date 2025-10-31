from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory, Scene, SensorConfig
import glob
import os
import numpy as np
import json
import argparse

class ExternalAgent(AbstractAgent):
    """External agent interface."""

    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        experiment_path: str = '/default/path/to/experiment',
    ):
        """
        Initializes the external agent object.
        :param trajectory_sampling: trajectory sampling specification
        :param experiment_path: path to the experiment results
        """
        self._trajectory_sampling = trajectory_sampling
        self.experiment_path = experiment_path

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

    def _read_xy_from_file(self, experiment_path: str, scene: Scene):
        """
        Reads a trajectory from a file.
        """
        scene_token = scene.scene_metadata.initial_token
        result_files = [f for f in glob.glob(f'{experiment_path}/*.json') if os.path.basename(f).startswith(scene_token)]
        assert len(result_files) == 1, f"Expected 1 result file, found {len(result_files)}"
        result_file = result_files[0]
        with open(result_file, 'r') as f:
            xy = np.array(json.load(f), dtype=np.float32)
        return xy

    def _read_xyyaw_from_file(self, experiment_path: str, scene: Scene):
        """
        Reads a trajectory from a file.
        """
        scene_token = scene.scene_metadata.initial_token
        result_files = [f for f in glob.glob(f'{experiment_path}/*.json') if os.path.basename(f).startswith(scene_token) and not os.path.basename(f).endswith('_agent_human_traj_diff.json')]
        assert len(result_files) == 1, f"Expected 1 result file, found {len(result_files)}"
        result_file = result_files[0]
        with open(result_file, 'r') as f:
            xyyaw = np.array(json.load(f), dtype=np.float32)
        return xyyaw

    def _integrate_xy_with_heading(self, xy, heading):
        # # pad heading with a 0 at the beginning
        # heading = np.vstack([np.zeros((1, 1)), heading])
        # # diff the absolute heading to get the relative heading
        # heading = np.diff(heading, axis=0)

        t1_to_t0 = self._convert_xy_heading_to_matrix(xy[0], heading[0].item())
        t2_to_t1 = self._convert_xy_heading_to_matrix(xy[1], heading[1].item())
        t3_to_t2 = self._convert_xy_heading_to_matrix(xy[2], heading[2].item())
        t4_to_t3 = self._convert_xy_heading_to_matrix(xy[3], heading[3].item())
        t5_to_t4 = self._convert_xy_heading_to_matrix(xy[4], heading[4].item())
        t6_to_t5 = self._convert_xy_heading_to_matrix(xy[5], heading[5].item())
        t7_to_t6 = self._convert_xy_heading_to_matrix(xy[6], heading[6].item())
        t8_to_t7 = self._convert_xy_heading_to_matrix(xy[7], heading[7].item())
        t2_to_t0 = t1_to_t0 @ t2_to_t1
        t3_to_t0 = t2_to_t0 @ t3_to_t2
        t4_to_t0 = t3_to_t0 @ t4_to_t3
        t5_to_t0 = t4_to_t0 @ t5_to_t4
        t6_to_t0 = t5_to_t0 @ t6_to_t5
        t7_to_t0 = t6_to_t0 @ t7_to_t6
        t8_to_t0 = t7_to_t0 @ t8_to_t7
        t2_xy, t2_heading = self._convert_matrix_to_xy_heading(t2_to_t0)
        t3_xy, t3_heading = self._convert_matrix_to_xy_heading(t3_to_t0)
        t4_xy, t4_heading = self._convert_matrix_to_xy_heading(t4_to_t0)
        t5_xy, t5_heading = self._convert_matrix_to_xy_heading(t5_to_t0)
        t6_xy, t6_heading = self._convert_matrix_to_xy_heading(t6_to_t0)
        t7_xy, t7_heading = self._convert_matrix_to_xy_heading(t7_to_t0)
        t8_xy, t8_heading = self._convert_matrix_to_xy_heading(t8_to_t0)
        return np.vstack([xy[0], t2_xy, t3_xy, t4_xy, t5_xy, t6_xy, t7_xy, t8_xy]), \
            np.vstack([heading[0], t2_heading, t3_heading, t4_heading, t5_heading, t6_heading, t7_heading, t8_heading])

    def _construct_trajectory(self, xy, future_headings):
        xy = xy.reshape(8, 2)
        # integrate xy along axis = 0
        xy, future_headings = self._integrate_xy_with_heading(xy, future_headings)
        # xy = np.cumsum(xy, axis=0)
        poses = np.concatenate([xy, future_headings], axis=1)
        return Trajectory(poses)

    def _convert_xy_heading_to_matrix(self, xy, theta):
        """
        Convert xy and heading to a 3x3 matrix
        """
        return np.array([[np.cos(theta), -np.sin(theta), xy[0]], [np.sin(theta), np.cos(theta), xy[1]], [0, 0, 1]])

    def _convert_matrix_to_xy_heading(self, matrix):
        """
        Convert a 3x3 matrix to xy and heading
        """
        return matrix[:2, 2], np.arctan2(matrix[1, 0], matrix[0, 0])

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param agent_input: Dataclass with agent inputs.
        :param scene: Current scene information.
        :return: Trajectory representing the predicted ego's position in future
        """
        experiment_path = self.experiment_path
        xyyaw = self._read_xyyaw_from_file(experiment_path, scene)
        xyyaw = xyyaw.reshape(8, 3)
        # xy = self._read_xy_from_file(experiment_path, scene)
        # xy = xy.reshape(8, 2)
        history_poses = scene.get_history_trajectory(4).poses
        prev_headings = history_poses[:, 2]
        t = np.arange(len(prev_headings))
        poly = np.polyfit(t, prev_headings, 2)
        
        # Predict 8 future headings
        future_t = np.arange(len(prev_headings), len(prev_headings) + 8)
        future_headings = np.polyval(poly, future_t).reshape(8, 1)

        xy = xyyaw[:, :2]
        future_headings = xyyaw[:, 2:]


        from get_rel_pose_from_abs_pose import calculate_relative_pose
        history_poses = calculate_relative_pose(history_poses).reshape(3, 3)
        history_x = history_poses[:, 0]
        history_y = history_poses[:, 1]
        history_yaw = history_poses[:, 2]

        future_x = history_x[-1].repeat(8).reshape(8, 1)
        future_y = history_y[-1].repeat(8).reshape(8, 1)
        future_yaw = history_yaw[-1].repeat(8).reshape(8, 1)

        # Fit quadratic polynomials separately for x and y coordinates
        # t = np.arange(len(history_x))
        # poly_x = np.polyfit(t, history_x, 2)
        # poly_y = np.polyfit(t, history_y, 2)
        # poly_yaw = np.polyfit(t, history_yaw, 2)
        
        # Predict 8 future xy points
        # future_t = np.arange(len(history_x), len(history_x) + 8)
        # future_x = np.polyval(poly_x, future_t).reshape(8, 1)
        # future_y = np.polyval(poly_y, future_t).reshape(8, 1)
        # future_yaw = np.polyval(poly_yaw, future_t).reshape(8, 1)

        # xy = np.hstack([future_x, xy[:, 1:2]])
        future_headings = future_yaw

        agent_trajectory = self._construct_trajectory(xy, future_headings)
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
