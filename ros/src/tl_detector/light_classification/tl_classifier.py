import numpy as np
import tensorflow as tf

import rospy
from styx_msgs.msg import TrafficLight

GRAPH_PATH = 'light_classification/model/frozen_inference_graph.pb'


class TLClassifier(object):
    def __init__(self):
        self.graph = self.load_graph(GRAPH_PATH)
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detect_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detect_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.detect_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            with tf.gfile.GFile(graph_file, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        return graph

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        image = np.expand_dims(image, axis=0)
        with self.graph.as_default():
            _, scores, classes, _ = self.sess.run(
                [self.detect_boxes, self.detect_scores,
                 self.detect_classes, self.num_detections],
                feed_dict={self.image_tensor: image})

        class_ = int(np.squeeze(classes)[0])
        score = np.squeeze(scores)[0]

        if score < .5:
            return TrafficLight.UNKNOWN

        if class_ == 1:
            state = TrafficLight.GREEN
        elif class_ == 2:
            state = TrafficLight.RED
        elif class_ == 3:
            state = TrafficLight.YELLOW

        return state
