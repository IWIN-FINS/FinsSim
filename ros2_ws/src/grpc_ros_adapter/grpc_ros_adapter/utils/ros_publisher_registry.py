import utils.ros_handle as rh
import time
class RosPublisherRegistry:

    _publishers = dict()

    @classmethod
    def get_publisher(cls, topic_name, data_class, queue_size=10):
        # TODO: better check and logging

        topic_name = rh.resolve_topic(topic_name)

        publisher = cls._publishers.get(topic_name, None)
        if not publisher:
            publisher = rh.Publisher(data_class, topic_name, qos_profile=queue_size)
            time.sleep(0.1)
            cls._publishers[topic_name] = publisher

        if hasattr(publisher, "data_class") and not publisher.data_class == data_class:
            rh.logerr(f"Publisher on topic {topic_name} cannot have data_class {data_class.__name__}")

        return publisher
