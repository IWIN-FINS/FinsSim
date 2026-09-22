
from protobuf.sensor_pb2 import PointCloud2
from protobuf import sensor_streaming_pb2
from protobuf import sensor_streaming_pb2_grpc
from utils import topic_streamer

class Context:
    pass

class SensorStreaming(sensor_streaming_pb2_grpc.SensorStreamingServicer):
    def __init__(self, callbacks={}):
        print("creating")
        self.publishers = {}
        self._callbacks = callbacks
        self._callback_contexts = {}


    def _get_callback_context(self, callback):
        context = self._callback_contexts.get(callback, None)
        if not context:
            context = Context()
        self._callback_contexts[callback] = context
        return context


    def trigger_callbacks(self, service_function, request):
        callbacks = self._callbacks.get(service_function.__name__, [])
        for c in callbacks:
            context = self._get_callback_context(c)
            c(request, context)



    def StreamCameraSensor(self, request_iterator, context):
        """
        Takes in a gRPC SensorStreamingRequest containing
        all the data needed to create and publish a sensor_msgs/Image
        ROS message.
        """
        for request in request_iterator:
            self.trigger_callbacks(self.StreamCameraSensor, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)

    def StreamLidarSensor(self, request_iterator, context):
        """
        Takes in a gRPC LidarStreamingRequest containing
        all the data needed to create and publish a PointCloud
        ROS message.
        """

        for request in request_iterator:
            self.trigger_callbacks(self.StreamLidarSensor, request)

        return sensor_streaming_pb2.LidarStreamingResponse(success=True)

    # def StreamRadarSensor(self, request, context):
    #     """
    #     Takes in a gRPC RadarStreamingRequest containing
    #     all the data needed to create and publish a RadarSpoke
    #     ROS message.
    #     """

    #     # TODO
    #     number_of_spokes = request.numSpokes

    #     for i in range(number_of_spokes):

    #         radar_spoke_msg = RadarSpoke()

    #         # Header
    #         header = std_msgs.msg.Header()
    #         header.frame_id = "milliampere_radar"
    #         header.stamp = rospy.Time.from_sec(request.timeInSeconds[i])
    #         # radar_spoke_msg.azimuth = request.azimuth[i]
    #         # radar_spoke_msg.intensity = request.radarSpokes[i * request.numSamples : i * request.numSamples + request.numSamples]

    #         # radar_spoke_msg.range_start = request.rangeStart
    #         # radar_spoke_msg.range_increment = request.rangeIncrement
    #         # radar_spoke_msg.min_intensity = request.minIntensity
    #         # radar_spoke_msg.max_intensity = request.maxIntensity
    #         # radar_spoke_msg.num_samples = request.numSamples

    #         self.radar_pub.publish(radar_spoke_msg)

    #     return sensor_streaming_pb2.RadarStreamingResponse(success=True)

    def StreamImuSensor(self, request_iterator, context):
        for request in request_iterator:
            self.trigger_callbacks(self.StreamImuSensor, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)


    def StreamPoseSensor(self, request_iterator, context):
        for request in request_iterator:
            self.trigger_callbacks(self.StreamPoseSensor, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)

    def StreamDepthSensor(self, request_iterator, context):
        for request in request_iterator:
            self.trigger_callbacks(self.StreamDepthSensor, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)

    def StreamDvlSensor(self, request_iterator, context):
        for request in request_iterator:
            self.trigger_callbacks(self.StreamDvlSensor, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)

    def StreamSonarImage(self, request_iterator, context):
        for request in request_iterator:
            self.trigger_callbacks(self.StreamSonarImage, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)

    def StreamSonarSensor(self, request_iterator, context):
        for request in request_iterator:
            self.trigger_callbacks(self.StreamSonarSensor, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)

    def StreamGnssSensor(self, request_iterator, context):
        for request in request_iterator:
            self.trigger_callbacks(self.StreamGnssSensor, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)

    def StreamAisSensor(self, request_iterator, context):
        for request in request_iterator:
            self.trigger_callbacks(self.StreamAisSensor, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)

    def StreamPointCloud(self, request_iterator, context):
        """
        Takes in a gRPC PointCloud2StreamingRequest containing
        all the data needed to create and publish a PointCloud2
        ROS message.
        """

        for request in request_iterator:
            self.trigger_callbacks(self.StreamPointCloud, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)

    def StreamPointCloud2(self, request_iterator, context):
        """
        Takes in a gRPC PointCloudStreamingRequest containing
        all the data needed to create and publish a PointCloud2
        ROS message.
        """

        for request in request_iterator:
            self.trigger_callbacks(self.StreamPointCloud2, request)

        return sensor_streaming_pb2.StreamingResponse(success=True)

    def RequestPointCloud2(self, request, context):
        from sensor_msgs.msg import PointCloud2 as PC2
        _streamer = topic_streamer.Streamer(lambda x: x, PC2)
        for msg in _streamer.start_stream(request, context):
            from protobuf.sensor_pb2 import PointCloud2
            response = sensor_streaming_pb2.PointCloud2StreamingRequest()
            response.data.height = msg.height
            response.data.width = msg.width
            response.data.isBigEndian = msg.is_bigendian
            response.data.pointStep = msg.point_step
            response.data.rowStep = msg.row_step
            response.data.is_dense = msg.is_dense

            response.data.header.timestamp = msg.header.stamp.to_sec()
            response.data.header.frameId = msg.header.frame_id

            response.data.data = msg.data
            from protobuf.sensor_pb2 import PointField
            fields = []
            for pf in msg.fields:
                field = PointField()
                field.name = pf.name
                field.offset = pf.offset
                field.datatype = pf.datatype
                field.count = pf.count
                fields.append(field)
            response.data.fields.extend(fields)

            callbacks = self._callbacks.get("RequestPointCloud2", [])
            for c in callbacks:
                c(response, context)
            yield response
