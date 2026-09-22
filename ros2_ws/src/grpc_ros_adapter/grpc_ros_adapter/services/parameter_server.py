from protobuf import parameter_server_pb2
from protobuf import parameter_server_pb2_grpc
from protobuf.std_pb2 import Empty

import utils.ros_handle as rh

class ParameterServer(parameter_server_pb2_grpc.ParameterServerServicer):

    def __init__(self):
        pass

    def GetParameter(self, request, context):
        param = rh.get_param(request.name, None)
        response = parameter_server_pb2.ParamValue()
        if isinstance(param, int):
            response.valueInt = param
        elif isinstance(param, bool):
            response.valueBool = param
        elif isinstance(param, float):
            response.valueDouble = param
        elif isinstance(param, str):
            response.valueStr = param
        elif param == None:
            pass
        else:
            raise Exception("Type of parameter not supported")
        
        return response


    def SetParameter(self, request, context):
        param = request.value
        which_one = param.WhichOneof("parameterValue")
        value = getattr(param, which_one, None)
        if which_one and value:
            rh.set_param(request.name, value)
        
        return Empty()
            
