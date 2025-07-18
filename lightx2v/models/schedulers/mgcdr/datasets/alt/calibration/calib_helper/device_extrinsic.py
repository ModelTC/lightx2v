#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:23:45 2024

@author: SENSETIME\zhuchengkai
"""
import os
import glob
import json
import numpy as np
import networkx as nx
from loguru import logger


class DeviceExtrinsic:

    def __init__(self, src, dst, extrinsic):
        self.src = src
        self.dst = dst
        self.extrinsic = extrinsic

    def __repr__(self, ):
        return str(self.__dict__)

    def transformPoints(self, objectPoints: np.ndarray) -> np.ndarray:
        # assert and reshape
        shape = list(objectPoints.shape)
        assert 3 == shape[-1]
        objectPoints = objectPoints.reshape((-1, 3))
        objectPoints = np.float64(objectPoints)
        objectPoints = np.insert(np.copy(objectPoints), 3, values=1, axis=1)

        # transform
        new_points = (self.extrinsic @ objectPoints.T).T
        new_points = new_points[:, :3]

        # reshape
        new_points = new_points.reshape(shape)
        return new_points


class DeviceExtrinsicGraph:

    @classmethod
    def _init_extrinsic_graph(cls, config_path: str) -> nx.DiGraph:
        graph = nx.DiGraph()
        extrinsic_file_list = sorted(glob.glob(
            os.path.join(config_path, "**", "*-extrinsic.json")
        ))
        for extrinsic_file in extrinsic_file_list:
            # load extrinsic file
            logger.debug(f"load extrinsic file: {extrinsic_file}")
            extrinsic_js_dict = json.load(open(extrinsic_file, "r"))
            assert len(extrinsic_js_dict) == 1
            extrinsic_object = list(extrinsic_js_dict.values())[0]
            assert "extrinsic" == extrinsic_object["param_type"]
            if os.path.basename(os.path.dirname(extrinsic_file)) not in [
                extrinsic_object["sensor_name"], extrinsic_object["target_sensor_name"]
            ]:
                logger.warning(f"skip unexpected file {extrinsic_file}")
                continue

            # record extrinsic as edges
            u_of_edge = extrinsic_object["sensor_name"]
            v_of_edge = extrinsic_object["target_sensor_name"]
            extrinsic = np.float64(
                extrinsic_object["param"]["sensor_calib"]["data"]
            )

            # path u_of_edge -> v_of_edge can not exist
            graph.add_node(u_of_edge)
            graph.add_node(v_of_edge)
            assert not graph.has_edge(u_of_edge, v_of_edge), f"duplicate defined: {u_of_edge} -> {v_of_edge}"  # noqa
            if nx.has_path(graph, u_of_edge, v_of_edge):
                shortest_path = nx.shortest_path(
                    graph,
                    source=u_of_edge,
                    target=v_of_edge,
                )
                logger.warning(
                    f"path {u_of_edge} -> {v_of_edge} exists: {shortest_path}"
                )

            # add edge
            graph.add_edge(
                u_of_edge,
                v_of_edge,
                extrinsic=np.copy(extrinsic),
            )
            # add reverse edge
            graph.add_edge(
                v_of_edge,
                u_of_edge,
                extrinsic=np.linalg.inv(extrinsic),
            )

        return graph

    def __init__(self, config_path: str):
        graph = self._init_extrinsic_graph(config_path)
        self.graph = graph

    def get_device_extrinsic(self, src: str, dst: str) -> DeviceExtrinsic:
        """
        Find shortest path and compute extrinsic from source sensor to target sensor.

        Parameters
        ----------
        src : str
            source sensor name.
        dst : str
            target sensor name.

        Returns
        -------
        DeviceExtrinsic
            .

        """
        extrinsic = np.eye(4, dtype=np.float64)
        path = nx.shortest_path(self.graph, source=src, target=dst)
        for u, v in zip(path[:-1], path[1:]):
            edge = self.graph.edges[u, v]["extrinsic"]
            extrinsic = edge @ extrinsic
        device_extrinsic = DeviceExtrinsic(
            src=src,
            dst=dst,
            extrinsic=extrinsic,
        )
        return device_extrinsic


if __name__ == "__main__":
    deg = DeviceExtrinsicGraph(
        config_path="test/data/2024_01_06_11_03_13_gacGtParser/calib"
    )

    nx.draw(deg.graph, with_labels=True)

    logger.info(deg.get_device_extrinsic(
        src="top_center_lidar",
        dst="center_camera_fov120",
    ))
    logger.info(deg.get_device_extrinsic(
        src="center_camera_fov120",
        dst="top_center_lidar",
    ))

    # transform test
    device_extrinsic = deg.get_device_extrinsic(
        src="car_center",
        dst="center_camera_fov120",
    )
    logger.info(device_extrinsic)
    points = np.random.random((2**16, 3))
    new_points = device_extrinsic.transformPoints(points)
