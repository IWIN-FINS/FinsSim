from functools import wraps, partial
import numpy as np

from std_msgs.msg import Float32, ColorRGBA
from geometry_msgs.msg import Vector3, Quaternion, Pose, Point
import protobuf.std_pb2 as std
import protobuf.geometry_pb2 as geometry

def add_method(cls, name=""):
    def decorator(func, name=""):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(self, *args, **kwargs)
        name = name or func.__name__
        setattr(cls, name, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return partial(decorator, name=name)

_UNITY_TO_ROS_TRANSFORM = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

_ROS_TO_UNITY_TRANSFORM = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

@add_method(Vector3, "as_msg")
def vector_as_msg(self):
    v = np.array([self.x, self.y, self.z, 1])
    # v = _ROS_TO_UNITY_TRANSFORM.dot(v)
    ret = geometry.Vector3()
    ret.x, ret.y, ret.z = v[0], v[1], v[2]
    return ret

@add_method(geometry.Vector3, "as_ros")
def vector_as_ros(self):
    v = np.array([self.x, self.y, self.z , 1])
    # v = _UNITY_TO_ROS_TRANSFORM.dot(v)
    ret = Vector3()
    ret.x, ret.y, ret.z = float(v[0]), float(v[1]), float(v[2])
    return ret

@add_method(Quaternion, "as_msg")
def quaternion_as_msg(self):
    ret = geometry.Quaternion()
    ret.x, ret.y, ret.z, ret.w = self.x, self.y, self.z, self.w
    return ret

@add_method(geometry.Quaternion, "as_ros")
def quaternion_as_ros(self):
    ret = Quaternion()
    ret.x = self.x
    ret.y = self.y
    ret.z = self.z
    ret.w = self.w
    return ret

@add_method(geometry.Point, "as_ros")
def point_as_ros(self):
    ret = Point()
    ret.x = self.x
    ret.y = self.y
    ret.z = self.z
    return ret

@add_method(Point, "as_msg")
def point_as_msg(self):
    ret = geometry.Point()
    ret.x, ret.y, ret.z = self.x, self.y, self.z
    return ret


@add_method(geometry.Pose, "as_ros")
def pose_as_ros(self):
    ret = Pose()
    ret.position = self.position.as_ros()
    ret.orientation = self.orientation.as_ros()
    return ret

@add_method(Pose, "as_msg")
def pose_as_msg(self):
    ret = geometry.Pose()
    ret.position.CopyFrom(self.position.as_msg())
    ret.orientation.CopyFrom(self.orientation.as_msg())
    return ret

@add_method(ColorRGBA, "as_msg")
def color_as_msg(self):
    ret = std.ColorRGBA()
    ret.r, ret.g, ret.b, ret.a = self.r, self.g, self.b, self.a
    return ret
