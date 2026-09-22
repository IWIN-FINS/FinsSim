import utils.ros_handle as rh
import time
from queue import Queue, Empty, Full

class Streamer:

    """
    Server streaming service.
    POC implementation
    """
    def __init__(self, make_response, topic_msg_type, callbacks={}):

        # map clients to their buffers
        self._registered_clients = {}
        # map address to clients that are listening to it
        self._address_to_clients_map = {}
        self._thread_sleep_if_empty = 0.05
        self._callbacks = callbacks

        self._topic_msg_type = topic_msg_type
        self._make_response = make_response
        self._message_counts = {}


    def _subscribe_to_topic(self, address, msg_type):
        def callback(msg, *args):
            clients = list(self._address_to_clients_map.get(address, []))
            count = self._message_counts.get(address, 0) + 1
            self._message_counts[address] = count
            if count <= 5 or count % 50 == 0:
                rh.loginfo(
                    f"gRPC stream relay received ROS message #{count} on {address} "
                    f"for {len(clients)} client(s)."
                )
            stale_clients = []
            for cl in clients:
                request_buffer = self._registered_clients.get((cl, address))
                if request_buffer is None:
                    stale_clients.append(cl)
                    continue
                try:
                    request_buffer.put(msg, block=False)
                except Full:
                    rh.logdebug(f"Dropping oldest remote-control message for {address}; client queue is full.")

            if stale_clients:
                active_clients = self._address_to_clients_map.get(address, [])
                self._address_to_clients_map[address] = [
                    cl for cl in active_clients if cl not in stale_clients
                ]

        rh.Subscription(msg_type, address, callback, 10)
        rh.loginfo(f"Subscribed gRPC stream relay to ROS topic {address}")

    @staticmethod
    def _normalize_address(address):
        address = address.lower()
        return rh.resolve_topic(address)

    def start_stream(self, request, context):
        
        address = self._normalize_address(request.address)
        request_buffer = Queue(100) # every client has a request buffer for every veh it controls 
        # context.peer() is stable across short reconnects from the same Unity
        # process. Use a per-stream key so a stale stream cannot unregister the
        # replacement stream that has the same peer string.
        client_id = (context.peer(), id(request_buffer))
        client_key = (client_id, address)

        # add buffer to registered clients
        self._registered_clients[client_key] = request_buffer

        # subscribe to topic if client is not already listening to it
        if address not in self._address_to_clients_map:
            self._address_to_clients_map[address] = [ client_id ]
            self._subscribe_to_topic(address, self._topic_msg_type)
        if client_id not in self._address_to_clients_map[address]:
            self._address_to_clients_map[address].append(client_id)
        rh.loginfo(
            f"Registered gRPC stream client {client_id} for ROS topic {address}; "
            f"active clients={len(self._address_to_clients_map[address])}."
        )

        while True:
            # if connection closed
            if not context.is_active():
                self._remove_client(request, client_id, request_buffer)
                return 

            if request_buffer.empty():
                time.sleep(self._thread_sleep_if_empty)
                continue

            data = None
            try:
                data = request_buffer.get(timeout=1)
            except Empty:
                continue

            try:
                response = self._make_response(data)
            except Exception as e:
                rh.logerr(f"Cannot create response message for topic in {address}. {repr(e)}.")
                continue

            yield response

    def _remove_client(self, request, client_id, request_buffer=None):
        address = self._normalize_address(request.address)
        client_key = (client_id, address)

        # Unity may recreate a server stream before the old stream observes
        # cancellation. Do not let the stale stream unregister the new buffer.
        if request_buffer is not None and self._registered_clients.get(client_key) is not request_buffer:
            rh.loginfo(f"Skip stale gRPC stream cleanup for ROS topic {address}; newer stream is active.")
            return

        cl_list = self._address_to_clients_map.get(address)
        if cl_list is not None and client_id in cl_list:
            cl_list.remove(client_id)
        self._registered_clients.pop(client_key, None)
        rh.loginfo(
            f"Removed gRPC stream client {client_id} for ROS topic {address}; "
            f"active clients={len(self._address_to_clients_map.get(address, []))}."
        )
