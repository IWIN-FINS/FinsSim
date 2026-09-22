#!/usr/bin/env python3

import roslib
import rospy
import unittest

import numpy as np
import roslaunch
from abc import abstractmethod, ABCMeta
from functools import partial


def wait_for(checker, timeout, step):
    def f():
        import time

        total = 0
        while total < timeout:
            done = checker()
            if done: 
                return True
            time.sleep(step)
            total += step
        return False
    return f

## A sample python unit test
class NodeTestHelper:

    def __init__(self, package_name, node_type, node_params=None):

        self.node_params = node_params or {}
        self.node_process = None
        self.ros_master = None
        self._all_processes = []
        # data received over subs callback
        self._data = {}
        # bool dict indicating wether writing to _data dict is done
        self._received_data = {}

        self.package_name = package_name
        self.node_type = node_type
        self.node_name = "".join(self.node_type.split('.')[:-1])

        self.test_name = "test_" + self.node_name
        self.configure_test()
        self._setup_test_class()

    def configure_test(self, **kwargs):
        """
        Override this method to set config parameters for test class
        """
        self.node_response_timeout = kwargs.get("node_response_timeout", 5)
        self.node_response_check_freq = kwargs.get("node_response_check_freq", 1)

    def subscribe(self, sub_name, typ):
        sub = rospy.Subscriber(sub_name, typ, self.set_subscriber_data(sub_name))

    @abstractmethod
    def setup_test(self):
        """
        Implement this method to set up the test case.
        Called only once on constructing test case, not for every test method call.
        """
        pass

    def launch_node(self, node):
        """
        Use this method to launch nodes for all your tests.
        Nodes are launched as subprocesses.
        At test finish, all subprocesses will be stopped automatically if 
        they are started by this method.
        Return handle to the newly opened process
        """
        process = self._launch.launch(node)
        self._all_processes.append(process)
        return process

    def get_subscriber_data(self, sub_name):
        if self.has_subscriber_received_data(sub_name):
            return self._data[sub_name]
        return None
    
    def has_subscriber_received_data(self, sub_name):
        return self._received_data.get(sub_name, False)

    def set_subscriber_data(self, sub_name):
        def set_data(data):
            # cls._data[sub_name] = object()
            self._data[sub_name] = data
            self._received_data[sub_name] = True
        return set_data

    def wait_for_response(self, sub_name, timeout=-1, step=-1):
        timeout = timeout if timeout >= 0 else self.node_response_timeout
        step = step if step >= 0 else self.node_response_check_freq

        wait_for_decorator = partial(wait_for, timeout=timeout, step=step)

        @wait_for_decorator
        def check_data():
            return self.has_subscriber_received_data(sub_name)
        
        return check_data()

    def _setup_test_class(self):

        if not self._is_ros_master_running():
            import subprocess
            self.ros_master = subprocess.Popen('roscore')
            self._all_processes.append(self.ros_master)
            rospy.sleep(1.5)

        self._launch = roslaunch.scriptapi.ROSLaunch()
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        rospy.init_node(uuid, anonymous=False)

        node = roslaunch.core.Node(self.package_name, self.node_type, respawn=True, name=self.node_name)
        for param_name in self.node_params:
            rospy.set_param(f"/{node.name}/{param_name}", self.node_params[param_name])

        self._launch.start()

        self.setup_test()

        # start node to test
        process = self.launch_node(node)

        self.node_process = process
        rospy.loginfo(f"{self.test_name} started")
        # rospy.sleep(1)

    @classmethod
    def _is_ros_master_running(cls):
        import rostopic
        try:
            # Checkif rosmaster is running or not.
            rostopic.get_topic_class('/rosout')
            return True
        except rostopic.ROSTopicIOException:
            return False


    def __exit__(self, p1, p2, p3):
        self.dispose()

    def __del__(self):
        self.dispose()

    def dispose(self):
        for proc in self._all_processes:
            if proc and proc.is_alive():
                proc.stop()
        
        if self._launch:
            self._launch.stop()

