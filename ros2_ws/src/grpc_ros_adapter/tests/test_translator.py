import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.joinpath("scripts")))

import msg_translator as msg_translator
import protobuf.sensor_pb2 as sensor_pb2
import protobuf.labust_pb2 as labust_pb2
import protobuf.marine_pb2 as marine_pb2

import geometry_msgs
import sensor_msgs


## setup
ros_msgs = msg_translator.get_all_ros_msgs()
proto_msgs = msg_translator.get_all_proto_msgs()
t = msg_translator.ProtoRosTranslator(proto_msgs, ros_msgs)

def test_proto2ros_obj():
    pose = labust_pb2.geometry__pb2.Point(x=1)
    trans = t.proto2ros(pose)
    assert trans.x == 1

def test_proto2ros_array():
    pose = labust_pb2.geometry__pb2.PoseWithCovariance()
    pose.covariance.extend([1, 2, 3, 4, 5, 6])
    trans = t.proto2ros(pose)
    assert trans.covariance == [1, 2, 3, 4, 5, 6]

def test_proto2ros_complex():
    pc = sensor_pb2.PointCloud()
    pc.points.extend([labust_pb2.geometry__pb2.Point(x=5, y=3), labust_pb2.geometry__pb2.Point(x=1, y=2)])
    trans = t.proto2ros(pc)
    assert trans.points[0].x == 5 and trans.points[1].y == 2

def test_ros2proto_obj():
    pr = geometry_msgs.msg.Point(x=5, y=3)
    trans = t.ros2proto(pr)
    assert trans.x == 5 and trans.y == 3

def test_ros2proto_array():
    pr = geometry_msgs.msg.PoseWithCovariance()
    pr.covariance = [1, 2, 3, 4, 5, 6]
    trans = t.ros2proto(pr)
    assert trans.covariance == [1, 2, 3, 4, 5, 6]

def test_ros2proto_complex():
    pc = sensor_msgs.msg.PointCloud()
    pc.points.extend([geometry_msgs.msg.Point(x=5, y=3), geometry_msgs.msg.Point(x=1, y=2)])
    trans = t.ros2proto(pc)
    assert trans.points[0].x == 5 and trans.points[1].y == 2