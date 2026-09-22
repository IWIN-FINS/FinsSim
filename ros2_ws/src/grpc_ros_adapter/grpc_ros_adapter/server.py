#!/usr/bin/env python3

from concurrent import futures
from pathlib import Path
import sys

_PACKAGE_DIR = Path(__file__).resolve().parent
for _path in (_PACKAGE_DIR, _PACKAGE_DIR / "protobuf"):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

import utils.ros_handle as rh
import grpc

from protobuf import ping_pb2_grpc
from protobuf import sensor_streaming_pb2_grpc
from protobuf import remote_control_pb2_grpc
from protobuf import tf_pb2_grpc
from protobuf import parameter_server_pb2_grpc
from protobuf import simulation_control_pb2_grpc
from protobuf import visualization_pb2_grpc

from services.ping_service import PingService
from services.sensor_streaming import SensorStreaming
from services.remote_control import RemoteControl
from services.parameter_server import ParameterServer
from services.frame_service import FrameService
from services.simulation_control import SimulationControl
from services.visualization import Visualization
from services.sensor_callbacks import *

def serve(server_ip, server_port):
    """
    Add service handles to server and start server execution
    """
    MAX_MESSAGE_LENGTH = 100 * 1024 * 1024
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=100),
    options=[
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ])


    ping_pb2_grpc.add_PingServicer_to_server(
            PingService(),
            server)

    sensor_streaming_callbacks =  {
        "StreamCameraSensor": [ publish_image ],
        "StreamSonarImage": [ publish_sonar_image ],
        "StreamImuSensor": [ publish_imu ],
        "StreamPoseSensor": [ publish_pose ],
        "StreamDepthSensor": [ publish_depth ],
        "StreamDvlSensor": [ publish_dvl ],
        "StreamSonarSensor": [ publish_sonar ],
        "StreamSonarFixSensor": [ publish_sonar_fix ],
        "StreamAisSensor" : [ publish_ais ],
        "StreamGnssSensor" : [ publish_gnss ],
        "StreamLidarSensor" : [publish_pointcloud],
        "StreamPointCloud2" : [publish_pointcloud2]
    }

    sensor_streaming_pb2_grpc.add_SensorStreamingServicer_to_server(
            SensorStreaming(sensor_streaming_callbacks),
            server)

    remote_control_pb2_grpc.add_RemoteControlServicer_to_server(
            RemoteControl(), server
    )

    tf_pb2_grpc.add_TfServicer_to_server(
            FrameService(), server
    )

    parameter_server_pb2_grpc.add_ParameterServerServicer_to_server(
            ParameterServer(), server
    )

    simulation_control_pb2_grpc.add_SimulationControlServicer_to_server(
            SimulationControl(), server
    )

    visualization_pb2_grpc.add_VisualizationServicer_to_server(
            Visualization(), server
    )

    server.add_insecure_port(server_ip + ':' + str(server_port))
    print(server_ip + ":" + str(server_port))
    server.start()
    rh.spin()
    server.stop(1)
#     server.wait_for_termination()


def main(args=None):
    rh.init('syntetic_data', args=args)
    server_ip = rh.get_param("~server_ip") or "[::]"
    server_port = rh.get_param("~server_port") or 30052
    topic_prefix = rh.get_param("~topic_prefix", "") or ""
    rh.set_topic_prefix(topic_prefix)
    if rh.get_topic_prefix():
        rh.loginfo(f"grpc_ros_adapter topic_prefix={rh.get_topic_prefix()}")
    serve(server_ip, server_port)


if __name__ == '__main__':
    main()
