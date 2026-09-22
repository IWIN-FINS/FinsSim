import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.joinpath("scripts")))

from node_test_helper import NodeTestHelper
import services.sensor_callbacks as callbacks
import rospy
import grpc
import pytest

from protobuf.sensor_streaming_pb2_grpc import SensorStreamingStub
from protobuf.sensor_streaming_pb2 import SonarStreamingRequest, DepthStreamingRequest
import protobuf.geometry_pb2 as geometry_proto

import geometry_msgs.msg as geometry_ros
import uuv_sensor_msgs.msg as uuv_sensor_ros

PACKAGE_NAME = "grpc_ros_adapter"
NODE_TYPE = "server.py"

@pytest.fixture(scope="module")
def node_test_helper():
    params = {
        "server_ip": "0.0.0.0",
        "server_port": 30052
    }
    helper = NodeTestHelper(PACKAGE_NAME, NODE_TYPE, params)
    rospy.sleep(2)
    yield helper
    # cleanup
    del helper

@pytest.fixture(scope="module")
def streaming_client():
    channel = grpc.insecure_channel('localhost:30052')
    client = SensorStreamingStub(channel)
    yield client
    # cleanup
    channel.close()
    
class TestServer:

    def test_depth(self, streaming_client, node_test_helper):
        sub_name = "/depth"
        node_test_helper.subscribe(sub_name, geometry_ros.PoseWithCovarianceStamped)
        pose = geometry_proto.PoseWithCovarianceStamped(\
            pose = geometry_proto.PoseWithCovariance(\
                pose = geometry_proto.Pose(\
                    position = geometry_proto.Point(x=1, z=-2))))
        request = DepthStreamingRequest(data=pose, address=sub_name)
        streaming_client.StreamDepthSensor(iter([request]))
        has_data = node_test_helper.wait_for_response(sub_name)
        assert has_data == True, "Response timed out..."
        data = node_test_helper.get_subscriber_data(sub_name)
        assert data.pose.pose.position.z == 2, "z should be 2"
        assert data.pose.pose.position.x == 0, "x should be 0, not set"

    # def test_sonar_fix(self, streaming_client, node_test_helper):
    #     sub_name = "/sonar_fix"
    #     node_test_helper.subscribe(sub_name, uuv_sensor_ros.SonarFix)
    #     request = SonarStreamingRequest(range=2, bearing=1, address=sub_name)
    #     streaming_client.StreamS(iter([request]))
    #     has_data = node_test_helper.wait_for_response(sub_name)
    #     assert has_data == True, "Response timed out..."
    #     data = node_test_helper.get_subscriber_data(sub_name)
    #     assert data.range == 2
    #     assert data.bearing == 1
