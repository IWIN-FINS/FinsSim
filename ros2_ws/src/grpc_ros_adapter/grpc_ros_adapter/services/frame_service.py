
import time

from protobuf import tf_pb2_grpc
from protobuf import tf_pb2
from protobuf.std_pb2 import Empty

import utils.ros_handle as rh
import utils.extensions
from tf2_msgs.msg import TFMessage
from threading import Lock
from geometry_msgs.msg import TransformStamped, Transform

class FrameService(tf_pb2_grpc.TfServicer):

    def __init__(self):
        self._static_tfs_lock = Lock()
        self._tfs_lock = Lock()
        self.static_tfs = []
        self.tfs = []
        self._has_new_data = False
        self._thread_sleep_time = 0.01
        rh.Subscription(TFMessage, "/tf", self.tf_callback, 10)
        rh.Subscription(TFMessage, "/tf_static", self.static_tf_callback, 10)

    def _find_tf_in_list(self, tf, l):
        return next(((i, v) for i, v in enumerate(l) \
            if v.child_frame_id == tf.child_frame_id \
            and v.header.frame_id == tf.header.frame_id), (-1, None))

    def _standardize_tf(self, tf):
        """
        Some tfs have '/' character at the beginning of the name. 
        It is removed
        """
        if tf.child_frame_id.startswith('/'):
            tf.child_frame_id = tf.child_frame_id[1:]
        if tf.header.frame_id.startswith('/'):
            tf.header.frame_id = tf.header.frame_id[1:]


    def tf_callback(self, message, *_args):
        with self._tfs_lock:
            for new_tf in message.transforms:
                self._standardize_tf(new_tf)
                idx, existing_tf = self._find_tf_in_list(new_tf, self.tfs)
                if existing_tf:
                    del self.tfs[idx]
                self.tfs.append(new_tf)
        self._has_new_data = True

    def static_tf_callback(self, message):
        with self._static_tfs_lock:
            for new_tf in message.transforms:
                self._standardize_tf(new_tf)
                _, existing_tf = self._find_tf_in_list(new_tf, self.static_tfs)
                if existing_tf:
                    del existing_tf
                self.static_tfs.append(new_tf)
        self._has_new_data = True

    def _create_tf_msg(self, tf):
        return tf_pb2.TfFrame(
            frameId=tf.header.frame_id,
            childFrameId=tf.child_frame_id,
            translation = tf.transform.translation.as_msg(),
            rotation = tf.transform.rotation.as_msg()
        )

    def _merge_frames_as_frame_list(self):
        frames = []
        with self._tfs_lock:
            for tf in self.tfs:
                tf_msg = self._create_tf_msg(tf)
                frames.append(tf_msg)

        with self._static_tfs_lock:
            for tf in self.static_tfs:
                tf_msg = self._create_tf_msg(tf)
                frames.append(tf_msg)

        tf_list = tf_pb2.TfFrameList()
        tf_list.frames.extend(frames)
        return tf_list

    def GetAllFrames(self, request, context):
        response = self._merge_frames_as_frame_list()
        return response


    def StreamAllFrames(self, request, context):
        while True:

            response = self._merge_frames_as_frame_list()
            self._has_new_data = False
            yield response

            time.sleep(self._thread_sleep_time)

    def PublishFrame(self, request_iterator, context):
        for request in request_iterator:
            frame = TransformStamped()
            frame.header.stamp = rh.Time.from_sec(request.header.timestamp)
            frame.header.frame_id = request.frameId
            frame.child_frame_id = request.childFrameId
            frame.transform = Transform()
            frame.transform.translation = request.translation.as_ros()
            frame.transform.rotation = request.rotation.as_ros()

            from utils.ros_publisher_registry import RosPublisherRegistry

            topic = request.address.lower()
            if topic.startswith("/"):
                topic = topic[1:]

            tfmsg = TFMessage()
            tfmsg.transforms.append(frame)
            pub = RosPublisherRegistry.get_publisher(topic, TFMessage)
            pub.publish(tfmsg)

        return Empty()
