# # Copyright 2022 Laboratory for Underwater Systems and Technologies (LABUST)
# # 
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# # 
# #     http://www.apache.org/licenses/LICENSE-2.0
# # 
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

from utils import topic_streamer
from protobuf import visualization_pb2
from protobuf import visualization_pb2_grpc
from protobuf import std_pb2
from visualization_msgs.msg import Marker, MarkerArray


class Visualization(visualization_pb2_grpc.VisualizationServicer):

    def __init__(self, callbacks={}):

        self._callbacks = callbacks
        self._streamer1 = topic_streamer.Streamer(
                Visualization.make_response, Marker)
        self._streamer2 = topic_streamer.Streamer(
                Visualization.make_response, MarkerArray)

    def SetMarker(self, request, context):

        for msg in self._streamer1.start_stream(request, context):
            try:
                response = Visualization.Ros2Msg(msg)
            except Exception as e:
                print(e)
            callbacks = self._callbacks.get("SetMarker", [])
            for c in callbacks:
                c(response, context)

            yield response

    def SetMarkerArray(self, request, context):
        for msg in self._streamer2.start_stream(request, context):

            response = visualization_pb2.MarkerArray()
            arr = []
            for marker in msg.markers:
                arr.append(Visualization.Ros2Msg(marker))
            response.markers = arr
            callbacks = self._callbacks.get("SetMarkerArray", [])
            for c in callbacks:
                c(response, context)

            yield response

    @staticmethod
    def make_response(response):
        return response

    @staticmethod
    def Ros2Msg(request):
        response = visualization_pb2.Marker()
        response.header.frameId = request.header.frame_id
        response.header.timestamp = request.header.stamp.to_nsec()

        response.ns = request.ns
        response.id = request.id
        response.type = request.type
        response.action = request.action
        response.pose.CopyFrom(request.pose.as_msg())
        response.scale.CopyFrom(request.scale.as_msg())
        response.color.CopyFrom(request.color.as_msg())
        response.lifetime = request.lifetime.to_sec()
        response.frameLocked = request.frame_locked
        response.points.extend([x.as_msg() for x in request.points])
        response.colors.extend([x.as_msg() for x in request.colors])
        response.text = request.text
        response.meshResource = request.mesh_resource
        response.meshUseEmbeddedMaterials = request.mesh_use_embedded_materials
        return response