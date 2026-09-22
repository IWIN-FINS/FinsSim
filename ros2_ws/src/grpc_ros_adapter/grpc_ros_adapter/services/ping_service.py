from protobuf import ping_pb2
from protobuf import ping_pb2_grpc

class PingService(ping_pb2_grpc.PingServicer):
    """
    Service used to send and receive dummy data to see if connection 
    is established
    """
    def __init__(self):
        pass

    def Ping(self, request, context):
        """
        Handles ping request
        """
        request.value = 1
        return request