import os


ROS_1 = 1
ROS_2 = 2

ROS_VERSION = int(os.getenv("ROS_VERSION"))
_HANDLER = None
if ROS_VERSION == ROS_1:
    import rospy
    _HANDLER = rospy
if ROS_VERSION == ROS_2:
    import rclpy
    _HANDLER = rclpy


IS_INIT = False
if not IS_INIT:
    _NODE = None

_TOPIC_PREFIX = ""


def _normalize_topic(topic):
    topic = str(topic or "").strip()
    if not topic:
        return "/"
    if not topic.startswith("/"):
        topic = "/" + topic
    while "//" in topic:
        topic = topic.replace("//", "/")
    return topic


def set_topic_prefix(prefix):
    global _TOPIC_PREFIX
    normalized = _normalize_topic(prefix)
    _TOPIC_PREFIX = "" if normalized == "/" else normalized.rstrip("/")


def get_topic_prefix():
    return _TOPIC_PREFIX


def resolve_topic(topic):
    normalized = _normalize_topic(topic)
    if not _TOPIC_PREFIX:
        return normalized
    if normalized == _TOPIC_PREFIX or normalized.startswith(_TOPIC_PREFIX + "/"):
        return normalized
    return _TOPIC_PREFIX + normalized

def init(node_name, namespace="", args=None):
    global _NODE
    global IS_INIT
    if ROS_VERSION == ROS_1:
        _NODE = _HANDLER.init_node(node_name, argv=args)
    if ROS_VERSION == ROS_2:
        _HANDLER.init(args=args)
        _NODE = _HANDLER.create_node(
            node_name,
            namespace=namespace,
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )
    IS_INIT = True
    return _NODE

def spin():
    if ROS_VERSION == ROS_1:
        _HANDLER.spin()
    if ROS_VERSION == ROS_2:
        _HANDLER.spin(_NODE)

def spin_once():
    if ROS_VERSION == ROS_1:
        _HANDLER.spin_once()
    if ROS_VERSION == ROS_2:
        _HANDLER.spin_once(_NODE)


def get_param(param_name:str, default=None):
    if ROS_VERSION == ROS_1:
        return _HANDLER.get_param(f"{param_name}", default)
    if ROS_VERSION == ROS_2:
        if param_name.startswith('~'):
            param_name = param_name[1:]
        p = _NODE.get_parameter_or(param_name, default)
        return getattr(p, "value", p)

def set_param(param_name, value):
    if ROS_VERSION == ROS_1:
        rospy.set_param(param_name, value)
    if ROS_VERSION == ROS_2:
        _NODE.set_parameters([rclpy.Parameter(param_name, value=value)])

class _Subscription:

    def __call__(
        self,
        msg_type,
        topic: str,
        callback,
        qos_profile: int,
        *args, **kwargs):
        if ROS_VERSION == ROS_1:
            return _HANDLER.Subscriber(topic, msg_type, callback, qos_profile)
        if ROS_VERSION == ROS_2:
            return _NODE.create_subscription(msg_type, topic, callback, qos_profile=qos_profile)

class _Publisher:

    def __call__(
        self,
        msg_type,
        topic: str,
        qos_profile: int
    ):
        if ROS_VERSION == ROS_1:
            return _HANDLER.Publisher(topic, msg_type, queue_size=qos_profile)
        if ROS_VERSION == ROS_2:
            return _NODE.create_publisher(msg_type, topic, qos_profile)


class _Time:

    def now(self):
        if ROS_VERSION == ROS_1:
            return _HANDLER.Time.now()
        if ROS_VERSION == ROS_2:
            return _NODE.get_clock().now().to_msg()

    def from_sec(self, sec):
        if ROS_VERSION == ROS_1:
            return _HANDLER.Time.from_sec(sec)
        if ROS_VERSION == ROS_2:
            return _HANDLER.time.Time(seconds=float(sec)).to_msg()


def logerr(msg, *args, **kwargs):
    if ROS_VERSION == ROS_1:
        _HANDLER.logerr(msg)
    if ROS_VERSION == ROS_2:
        _NODE.get_logger().error(msg)

def loginfo(msg):
    if ROS_VERSION == ROS_1:
        _HANDLER.loginfo(msg)
    if ROS_VERSION == ROS_2:
        _NODE.get_logger().info(msg)

def logdebug(msg):
    if ROS_VERSION == ROS_1:
        _HANDLER.logdebug(msg)
    if ROS_VERSION == ROS_2:
        _NODE.get_logger().debug(msg)

Subscription = _Subscription()
Publisher = _Publisher()
Time = _Time()

__all__ = [
    "Subscription",
    "Publisher",
    "Time",
    "get_param",
    "set_param",
    "set_topic_prefix",
    "get_topic_prefix",
    "resolve_topic",
]
