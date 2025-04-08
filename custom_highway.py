from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.behavior import IDMVehicle

Observation = np.ndarray


class MyHighwayEnv(AbstractEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"}, # Kinematics observation
                "action": {
                    "type": "DiscreteMetaAction", # Discrete action space with meta actions
                },
                "lanes_count": 2,  # 2 lanes for overtaking
                "vehicles_count": 2,  # 2 obstacle vehicles
                "controlled_vehicles": 1,  # 1 vehicle controlled by the RL agent
                "initial_lane_id": 1,  # ego vehicle start in right lane
                "duration": 10,  # [s]
                "ego_spacing": 2,  # spacing in between
                "vehicles_density": 1,  # density of vehicles
                "collision_reward": -1,  # reward when colliding with other vehicle
                "right_lane_reward": 0.4,  # reward when driving on the right
                "high_speed_reward": 0.4, # reward when driving at high speed
                "lane_change_reward": 0,  # reward at each lane change action
                "reward_speed_range": [25, 35], # speed range for reward calculation
                "normalize_reward": True, # normalize the reward
                "offroad_terminal": True, # terminate if offroad
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=40
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Set up vehicles for the overtaking scenario."""

        # Define initial and overtaking lane indices
        ego_lane_id = self.config["initial_lane_id"]
        overtaking_lane_id = ego_lane_id - 1

        # Define lane indices as tuples
        ego_lane_index = ("0", "1", ego_lane_id)
        overtaking_lane_index = ("0", "1", overtaking_lane_id)

        # Get lane objects using the correct indices
        ego_lane = self.road.network.get_lane(ego_lane_index)
        overtaking_lane = self.road.network.get_lane(overtaking_lane_index)

        # Ego vehicle
        ego_vehicle = self.action_type.vehicle_class(
            road=self.road,
            position=ego_lane.position(25, 0),
            heading=ego_lane.heading_at(25),
            speed=25)
        self.controlled_vehicles = [ego_vehicle]
        self.road.vehicles.append(ego_vehicle)

        # Preceding vehicle in the same lane, ahead of the ego vehicle
        preceding_vehicle = IDMVehicle(
            road=self.road,
            position=ego_lane.position(45, 0),
            heading=ego_lane.heading_at(45),
            speed=20)
        self.road.vehicles.append(preceding_vehicle)

        # Approaching vehicle in the overtaking lane, behind the ego vehicle
        approaching_vehicle = IDMVehicle(
            road=self.road,
            position=overtaking_lane.position(0, 0),
            heading=overtaking_lane.heading_at(0),
            speed=35)
        self.road.vehicles.append(approaching_vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        if self.config["normalize_reward"]:
            reward = utils.lmap(
                reward,
                [
                    self.config["collision_reward"],
                    self.config["high_speed_reward"] + self.config["right_lane_reward"],
                ],
                [0, 1],
            )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

