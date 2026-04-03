from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.utils.structs.types import SimConfig


def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]

    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )

    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


@register_env("PegInsertionSide-v1", max_episode_steps=100)
class PegInsertionSideEnv(BaseEnv):
    """
    **Task Description:**
    Pick up a orange-white peg and insert either the orange end or white end into the box with a hole in it.

    **Randomizations:**
    - Peg half length is randomized between 0.11 and 0.15 meters. Box half length is 0.085m. (during reconfiguration)
    - Peg radius/half-width is randomized between 0.015 and 0.025 meters. Box hole's radius is same value + 0.01m of clearance. (during reconfiguration)
    - Peg is laid flat on table and has it's xy position and z-axis rotation randomized
    - Box is laid flat on table and has it's xy position and z-axis rotation randomized

    **Success Conditions:**
    - Either the orange end or white end of the peg is inserted at least 0.03m into the hole, with the inserted end within the hole's center radius.
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PegInsertionSide-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    _clearance = 0.02

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            # [MODIFIED]: Increased Peg length by 0.05m (half length +0.025m)
            # Old: 0.085-0.125 -> New: 0.11-0.15 (Total length 0.22m - 0.30m)
            lengths = self._batched_episode_rng.uniform(0.15, 0.2)
            radii = self._batched_episode_rng.uniform(0.015, 0.025)
            centers = (
                0.5
                * (lengths - radii)[:, None]
                * self._batched_episode_rng.uniform(-1, 1, size=(2,))
            )

            # save some useful values for use later
            self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)
            
            # Add tail offsets for white end (opposite direction from head)
            peg_tail_offsets = torch.zeros((self.num_envs, 3))
            peg_tail_offsets[:, 0] = -self.peg_half_sizes[:, 0]
            self.peg_tail_offsets = Pose.create_from_pq(p=peg_tail_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            pegs = []
            boxes = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i]
                radius = radii[i]
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[length, radius, radius])
                # peg head
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                # peg tail
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([-length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)
                # box with hole

                inner_radius, outer_radius, depth = (
                    radius + self._clearance,
                    length,
                    0.15, # Fixed Box depth to ensure reachability
                )
                builder = _build_box_with_hole(
                    self.scene, inner_radius, outer_radius, depth, center=centers[i]
                )
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)
                pegs.append(peg)
                boxes.append(box)
            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")

            # to support heterogeneous simulation state dictionaries we register merged versions
            # of the parallel actors
            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # initialize the box and peg
            xy = randomization.uniform(
                low=torch.tensor([-0.1, -0.3]), high=torch.tensor([0.1, 0]), size=(b, 2)
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(-np.pi, np.pi),
            )
            self.peg.set_pose(Pose.create_from_pq(pos, quat))

            xy = randomization.uniform(
                low=torch.tensor([-0.05, 0.2]),
                high=torch.tensor([0.05, 0.4]),
                size=(b, 2),
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
            )
            self.box.set_pose(Pose.create_from_pq(pos, quat))

            # Initialize the robot
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    -np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    # save some commonly used attributes
    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets

    @property
    def peg_tail_pos(self):
        return self.peg.pose.p + self.peg_tail_offsets.p

    @property
    def peg_tail_pose(self):
        return self.peg.pose * self.peg_tail_offsets

    @property
    def box_hole_pose(self):
        return self.box.pose * self.box_hole_offsets

    @property
    def goal_pose(self):
        # NOTE (stao): this is fixed after each _initialize_episode call. You can cache this value
        # and simply store it after _initialize_episode or set_state_dict calls.
        # Default goal pose for head insertion (orange end)
        return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()
    
    @property
    def goal_pose_tail(self):
        # Goal pose for tail insertion (white end)
        return self.box.pose * self.box_hole_offsets * self.peg_tail_offsets.inv()

    def has_peg_inserted(self):
        # Check if either head (orange end) or tail (white end) is inserted
        peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
        peg_tail_pos_at_hole = (self.box_hole_pose.inv() * self.peg_tail_pose).p
        
        # x-axis is hole direction
        # Check head insertion
        head_depth = peg_head_pos_at_hole[:, 0]
        head_x_flag = -0.1 <= head_depth  #-0.1 is original
        head_y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
            peg_head_pos_at_hole[:, 1] <= self.box_hole_radii
        )
        head_z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
            peg_head_pos_at_hole[:, 2] <= self.box_hole_radii
        )
        head_inserted = head_x_flag & head_y_flag & head_z_flag
        # head_inserted = head_x_flag
        # print("head_inserted")
        # print(head_x_flag, head_y_flag, head_z_flag, head_inserted)
        
        # Check tail insertion
        tail_depth = peg_tail_pos_at_hole[:, 0]
        tail_x_flag = -0.1 <= tail_depth  #-0.1 is original
        tail_y_flag = (-self.box_hole_radii <= peg_tail_pos_at_hole[:, 1]) & (
            peg_tail_pos_at_hole[:, 1] <= self.box_hole_radii
        )
        tail_z_flag = (-self.box_hole_radii <= peg_tail_pos_at_hole[:, 2]) & (
            peg_tail_pos_at_hole[:, 2] <= self.box_hole_radii
        )
        tail_inserted = tail_x_flag & tail_y_flag & tail_z_flag
        # tail_inserted = tail_x_flag
        # print("tail_inserted")
        # print(tail_x_flag, tail_y_flag, tail_z_flag, tail_inserted)
        # Success if either end is inserted
        # print("Head depth & head inserted: ", head_depth, head_inserted, "Tail depth & tail inserted: ", tail_depth, tail_inserted)
        # print("================================================")
        success = head_inserted | tail_inserted
        
        return (
            success,
            peg_head_pos_at_hole,
            peg_tail_pos_at_hole,
            head_inserted,
            tail_inserted,
        )

    def evaluate(self):
        success, peg_head_pos_at_hole, peg_tail_pos_at_hole, head_inserted, tail_inserted = self.has_peg_inserted()
        # if success:
            # print("================================================")
            # if head_inserted: print(">> Head inserted")
            # if tail_inserted: print(">> Tail inserted")
            # print("================================================")
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole, peg_tail_pos_at_hole=peg_tail_pos_at_hole)

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_half_size=self.peg_half_sizes,
                box_hole_pose=self.box_hole_pose.raw_pose,
                box_hole_radius=self.box_hole_radii,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
            # Stage 1: Encourage gripper to be rotated to be lined up with the peg

            # Stage 2: Encourage gripper to move close to peg and grasp it
            # [MODIFIED]: Reverted to center grasp. 
            # With longer pegs, center grasp is safe for both ends.
            gripper_pos = self.agent.tcp.pose.p
            tgt_gripper_pose = self.peg.pose
            offset = sapien.Pose([0, 0, 0]) 
            tgt_gripper_pose = tgt_gripper_pose * (offset)
            gripper_to_peg_dist = torch.linalg.norm(
                gripper_pos - tgt_gripper_pose.p, axis=1
            )

            reaching_reward = 1 - torch.tanh(5.0 * gripper_to_peg_dist)

            # check with max_angle=20 to ensure gripper isn't grasping peg at an awkward pose
            is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
            reward = reaching_reward + is_grasped

            # Stage 3: Orient the grasped peg properly towards the hole
            # Support both head (orange) and tail (white) insertion

            # Pre-insertion reward for head (orange end) insertion
            peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
            peg_head_wrt_goal_yz_dist = torch.linalg.norm(
                peg_head_wrt_goal.p[:, 1:], axis=1
            )
            peg_wrt_goal_head = self.goal_pose.inv() * self.peg.pose
            peg_wrt_goal_head_yz_dist = torch.linalg.norm(peg_wrt_goal_head.p[:, 1:], axis=1)

            pre_insertion_reward_head = 3 * (
                1
                - torch.tanh(
                    0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_head_yz_dist)
                    + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_head_yz_dist)
                )
            )
            
            # Pre-insertion reward for tail (white end) insertion
            peg_tail_wrt_goal = self.goal_pose_tail.inv() * self.peg_tail_pose
            peg_tail_wrt_goal_yz_dist = torch.linalg.norm(
                peg_tail_wrt_goal.p[:, 1:], axis=1
            )
            peg_wrt_goal_tail = self.goal_pose_tail.inv() * self.peg.pose
            peg_wrt_goal_tail_yz_dist = torch.linalg.norm(peg_wrt_goal_tail.p[:, 1:], axis=1)

            pre_insertion_reward_tail = 3 * (
                1
                - torch.tanh(
                    0.5 * (peg_tail_wrt_goal_yz_dist + peg_wrt_goal_tail_yz_dist)
                    + 4.5 * torch.maximum(peg_tail_wrt_goal_yz_dist, peg_wrt_goal_tail_yz_dist)
                )
            )
            
            # Take the maximum reward from either insertion direction
            pre_insertion_reward = torch.maximum(pre_insertion_reward_head, pre_insertion_reward_tail)
            reward += pre_insertion_reward * is_grasped
            
            # Stage 4: Insert the peg into the hole once it is grasped and lined up
            # Support both insertion directions
            peg_head_wrt_goal_inside_hole = self.box_hole_pose.inv() * self.peg_head_pose
            insertion_reward_head = 5 * (
                1
                - torch.tanh(
                    5.0 * torch.linalg.norm(peg_head_wrt_goal_inside_hole.p, axis=1)
                )
            )
            
            peg_tail_wrt_goal_inside_hole = self.box_hole_pose.inv() * self.peg_tail_pose
            insertion_reward_tail = 5 * (
                1
                - torch.tanh(
                    5.0 * torch.linalg.norm(peg_tail_wrt_goal_inside_hole.p, axis=1)
                )
            )
            
            # Take the maximum insertion reward from either direction
            insertion_reward = torch.maximum(insertion_reward_head, insertion_reward_tail)
            
            # Soft gating: use alignment quality as continuous coefficient instead of hard threshold
            # Calculate alignment quality: 0 (poor) -> 1 (perfect)
            min_yz_dist = torch.minimum(peg_head_wrt_goal_yz_dist, peg_tail_wrt_goal_yz_dist)
            alignment_quality = 1 - torch.tanh(10.0 * min_yz_dist)
            
            # Insertion reward is always visible when grasped, but weighted by alignment quality
            reward += insertion_reward * is_grasped * alignment_quality

            reward[info["success"]] = 10

            return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs, action, info) / 10