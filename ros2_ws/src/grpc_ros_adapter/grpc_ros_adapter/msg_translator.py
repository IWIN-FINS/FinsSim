import rosmsg, genpy
import roslib.message as rosmessage
import std_msgs
import rospkg
import rospy

import google.protobuf as protobuf
import google.protobuf.descriptor as descriptor

from copy import deepcopy


from geometry_msgs.msg import Pose

class ProtoRosTranslator:
    """
    Class made for translating between generic ros/proto files

    Translator matches messages by name (full name) in given ros/proto messages

    """

    def __init__(self, proto_msg_list, ros_msg_list, 
            ros2proto_custom_translator=None, proto2ros_custom_translator=None):
        self.ros_msgs = dict(zip((msg._type for msg in ros_msg_list), ros_msg_list))
        self.proto_msgs = dict(zip((msg.full_name for msg in proto_msg_list), proto_msg_list))
        self.ros2proto_custom_translator = ros2proto_custom_translator or {}
        self.proto2ros_custom_translator = proto2ros_custom_translator or {}


    def add_proto_msgs(self, proto_msgs):
        """
        Insert new proto message(s) to the proto dictionary.

        NOTE 1: Accepts both list and single MessagesDescriptor objects
        """
        if isinstance(proto_msgs, list):
            self.proto_msgs.update(zip((msg.full_name for msg in proto_msgs), proto_msgs))
            return
        self.proto_msgs[proto_msgs.full_name] = proto_msgs

    def add_ros_msgs(self, ros_msgs):
        """
        Insert new ros message(s) to the ros dictionary.

        NOTE 1: Accepts both list and single genpy.Message objects
        """
        if isinstance(ros_msgs, list):
            self.ros_msgs.update(zip((msg._type for msg in ros_msgs), ros_msgs))
            return
        self.ros_msgs[ros_msgs._type] = ros_msgs


    def is_proto_msg(self, msg):
        """
        Is given object a protobuf message
        """
        return isinstance(msg, protobuf.message.Message)

    def is_ros_msg(self, msg):
        """
        Is given object a ros message
        """
        return isinstance(msg, genpy.Message)


    def ros2proto(self, msg):
        """
        Given a ros message, create new proto message and fill its contents
        by field names
        """
        if not self.is_ros_msg(msg):
            rospy.logdebug(f"Given object {msg} is not a ros message")
            return None

        custom_trans = self.ros2proto_custom_translator.get(msg._type, None)
        if custom_trans:
            return custom_trans(msg)

        proto_msg = self.get_proto_msg_cls_for_ros_msg(msg)
        new_proto = self.create_empty_proto(proto_msg)

        for field_info in new_proto.DESCRIPTOR.fields:
            self._set_ros_field2proto_field(field_info, msg, new_proto)
        return new_proto

    def proto2ros(self, msg):
        """
        Given a proto message, create new ros message and fill its contents
        by field names
        """
        if not self.is_proto_msg(msg):
            rospy.logdebug(f"Given object {msg} is not a proto message")
            return None

        custom_trans = self.proto2ros_custom_translator.get(msg.DESCRIPTOR.full_name, None)
        if custom_trans:
            return custom_trans(msg)

        ros_msg = self.get_ros_msg_cls_for_proto_msg(msg)
        new_ros_msg = self.create_empty_ros(ros_msg)

        for field_info in msg.DESCRIPTOR.fields:
            self._set_proto_field2ros_field(field_info, msg, new_ros_msg)
        return new_ros_msg

    def create_empty_proto(self, proto_msg):
        """
        Create new proto message from given class
        """
        constructor = protobuf.reflection.MakeClass(proto_msg)
        return constructor()

    def create_empty_ros(self, ros_msg):
        """
        Create new ros message from given class
        """
        return ros_msg()

    def get_proto_msg_cls_for_ros_msg(self, ros_msg):
        ros_msg_name = ros_msg._type
        replace_slashes = ros_msg_name.replace("/", ".")
        maybe_msg = self.proto_msgs.get(replace_slashes, None)
        if maybe_msg:
            return maybe_msg

        split = replace_slashes.split(".")
        new_try_path_list = [(name.replace("_msgs", ".") if i != len(split)-1 else name) for (i, name) in enumerate(split) ]
        new_try_path = "".join(new_try_path_list)
        maybe_msg = self.proto_msgs.get(new_try_path, None)
        return maybe_msg

    def get_ros_msg_cls_for_proto_msg(self, proto_msg):

        proto_msg_name = proto_msg.DESCRIPTOR.full_name
        replace_dots = proto_msg_name.replace(".", "/")
        maybe_msg = self.ros_msgs.get(replace_dots, None)
        if maybe_msg:
            return maybe_msg

        split = replace_dots.split("/")
        new_try_path_list = [(name + "_msgs/" if i != len(split)-1 else name) for (i, name) in enumerate(split) ]
        new_try_path = "".join(new_try_path_list)
        maybe_msg = self.ros_msgs.get(new_try_path, None)
        return maybe_msg


    def _set_proto_field2ros_field(self, field_info, proto_msg, ros_msg):
        # typ = field_info.message_type
        if isinstance(ros_msg, std_msgs.msg.Header):
            ros_msg.seq = 0
            ros_msg.stamp.secs = int(proto_msg.timestamp)
            ros_msg.stamp.nsecs = int((proto_msg.timestamp - ros_msg.stamp.secs) * 1e9)
            ros_msg.frame_id = proto_msg.frameId
        elif isinstance(ros_msg, rospy.Time):
            ros_msg.secs = proto_msg.secs
            ros_msg.nsecs = proto_msg.nsecs
        elif isinstance(ros_msg, rospy.Duration):
            ros_msg.secs = proto_msg.secs
            ros_msg.nsecs = proto_msg.nsecs
        else:
            name = field_info.name
            setattr(ros_msg, name, self._proto2ros_token(field_info, getattr(proto_msg, name)))

    def _set_ros_field2proto_field(self, field_info, ros_msg, proto_msg):
        if isinstance(ros_msg, std_msgs.msg.Header):
            proto_msg.timestamp = ros_msg.stamp.secs + 1e-9 * ros_msg.stamp.nsecs
            proto_msg.frameId = ros_msg.frame_id
        elif isinstance(ros_msg, rospy.Time):
            proto_msg.secs = ros_msg.secs
            proto_msg.nsecs = ros_msg.nsecs
        elif isinstance(ros_msg, rospy.Duration):
            proto_msg.secs = ros_msg.secs
            proto_msg.nsecs = ros_msg.nsecs
        else:
            name = field_info.name
            field = getattr(proto_msg, name)
            ros_value = getattr(ros_msg, name)
            if self.is_ros_primitive(ros_value):
                value = self._ros2proto_primitive(ros_value)
                setattr(proto_msg, name, value)
                return
            elif self.is_ros_list(ros_value):
                field.extend(self._ros2proto_list(ros_value))
                return
            elif self.is_ros_obj(ros_value):
                field.CopyFrom(self._ros2proto_obj(ros_value))
                return 
            raise Exception("Error while translating proto to ros message!")
        pass

    def _ros2proto_list(self, ros_list):
        ret_list = []
        for item in ros_list:
            if self.is_ros_list(item):
                ret_list.append(self._ros2proto_list(item))
            elif self.is_ros_obj(item):
                ret_list.append(self._ros2proto_obj(item))
            elif self.is_ros_primitive(item):
                ret_list.append(self._ros2proto_primitive(item))
        return ret_list

    def _proto2ros_list(self, field_info, proto_list):
        ret_list = []
        for item in proto_list:
            ret_list.append(self._proto2ros_token(field_info, item, from_list=True))
        return ret_list

    def _ros2proto_obj(self, ros_obj):
        return self.ros2proto(ros_obj)

    def _proto2ros_obj(self, proto_obj):
        return self.proto2ros(proto_obj)

    def _ros2proto_primitive(self, ros_primitive):
        return ros_primitive

    def _proto2ros_primitive(self, field_info, proto_primitive):
        if field_info.type == descriptor.FieldDescriptor.TYPE_INT32 or \
                field_info.type == descriptor.FieldDescriptor.TYPE_INT64 or \
                field_info.type == descriptor.FieldDescriptor.TYPE_UINT32 or \
                field_info.type == descriptor.FieldDescriptor.TYPE_UINT64:
            return int(proto_primitive)
            
        if field_info.type == descriptor.FieldDescriptor.TYPE_FLOAT or \
                field_info.type == descriptor.FieldDescriptor.TYPE_DOUBLE:
            return float(proto_primitive)
        if field_info.type == descriptor.FieldDescriptor.TYPE_STRING:
            return str(proto_primitive)
        raise Exception("Unsupported primitive type in proto!")

    def _proto2ros_token(self, field_info, proto_value, from_list=False):

        if self.is_proto_primitive(field_info, from_list):
            return self._proto2ros_primitive(field_info, proto_value)
        elif self.is_proto_list(field_info, from_list):
            return self._proto2ros_list(field_info, proto_value)
        elif self.is_proto_obj(field_info, from_list):
            return self._proto2ros_obj(proto_value)
        
        raise Exception("Error while translating proto to ros message!")

    def is_ros_list(self, ros_value):
        return isinstance(ros_value, list)

    def is_ros_obj(self, ros_value):
        return self.is_ros_msg(ros_value)

    def is_ros_primitive(self, ros_value):
        return isinstance(ros_value, float) or \
                isinstance(ros_value, int) or \
                isinstance(ros_value, str)

    def is_proto_list(self, field_info, from_list=False):
        return not from_list and field_info.label == descriptor.FieldDescriptor.LABEL_REPEATED

    def is_proto_obj(self, field_info, from_list=False):
        return field_info.type == descriptor.FieldDescriptor.TYPE_MESSAGE \
            and (field_info.label != descriptor.FieldDescriptor.LABEL_REPEATED or from_list)

    def is_proto_primitive(self, field_info, from_list=False):
        return field_info.type in [
            descriptor.FieldDescriptor.TYPE_INT32,
            descriptor.FieldDescriptor.TYPE_INT64,
            descriptor.FieldDescriptor.TYPE_UINT32,
            descriptor.FieldDescriptor.TYPE_UINT64,
            descriptor.FieldDescriptor.TYPE_FLOAT,
            descriptor.FieldDescriptor.TYPE_DOUBLE,
            descriptor.FieldDescriptor.TYPE_BOOL,
            descriptor.FieldDescriptor.TYPE_STRING
        ] \
        and (field_info.label != descriptor.FieldDescriptor.LABEL_REPEATED or from_list)



def get_all_ros_msgs():
    ros_msgs = []
    pack = rospkg.RosPack()
    for p in pack.list():
        for msg_name in rosmsg.list_msgs(p, pack):
            msg = rosmessage.get_message_class(msg_name)
            if msg:
                ros_msgs.append(msg)
    return ros_msgs

def get_all_proto_msgs():
    database = protobuf.symbol_database.Default()
    proto_msgs = database._classes
    return proto_msgs
