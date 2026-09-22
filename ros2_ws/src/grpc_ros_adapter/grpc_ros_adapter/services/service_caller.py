from protobuf import commander_service_pb2
from protobuf import commander_service_pb2_grpc


import rospy



class ServiceCaller(commander_service_pb2_grpc.CommanderServicer):
    """
    NOTE: Used for a specific use case. This is not tested and not to be used!!

    Service used to control the flow of the simulation
    Simulator dictates the clock and does a Step request for ROS nodes
    """
    def __init__(self):
        self.velcon_selection = rospy.Publisher('d2/velocity_selection', Int32MultiArray, queue_size=1)
        pass

    def PrimitivePointer(self, request, context):
        # temp ros1 solution
        # ROS2 has request and response defined inside class
        from std_msgs.msg import Int32MultiArray
        from labust_msgs.srv import PointerPrimitiveService, PointerPrimitiveServiceRequest
        rospy.wait_for_service('/d2/commander/primitive/pointer')
        try:
            pointer_service = rospy.ServiceProxy('/d2/commander/primitive/pointer', PointerPrimitiveService)
            srvout = PointerPrimitiveServiceRequest()
            srvout.radius = request.radius
            srvout.radius_topic = request.radiusTopic
            srvout.guidance_topic = request.guidanceTopic
            srvout.fov_guidance = request.fovGuidance
            srvout.guidance_enable = request.guidanceEnable
            srvout.guidance_target = request.guidanceTarget.as_ros()
            srvout.vertical_offset = request.verticalOffset
            resp = pointer_service(srvout)
            outselect = Int32MultiArray()
            outselect.data = [2,2,2,2,2,2]
            self.velcon_selection.publish(outselect)
            return commander_service_pb2.PrimitivePointerResponse(success=True)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)       
            return commander_service_pb2.PrimitivePointerResponse(success=False)
