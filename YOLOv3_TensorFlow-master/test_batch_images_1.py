import tensorflow as tf
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
class YoloV3:
    """Class to load ssd model and run inference."""
    INPUT_NAME = 'Placeholder:0'
    BOXES_NAME = 'output/boxes:0'
    CLASSES_NAME = 'output/labels:0'
    SCORES_NAME = 'output/scores:0'
    #NUM_DETECTIONS_NAME = 'output/num_detections:0'
    def __init__(self, frozen_graph):
        self.graph = tf.Graph()
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)

        if graph_def is None:
              raise RuntimeError('Cannot find inference graph.')

        with self.graph.as_default():
              tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)
    def run(self, image):
        """
        image should be normalized to [0,1] and RGB order
        """
        boxes, classes, scores = self.sess.run(
                    [self.BOXES_NAME, self.CLASSES_NAME, self.SCORES_NAME],
                    feed_dict={self.INPUT_NAME: image})
        return boxes, classes.astype(np.int64), scores

if __name__ == '__main__':
    import os
    import glob
    import numpy as np
    import cv2

    from utils.plot_utils import get_color_table, plot_one_box
    from utils.misc_utils import parse_anchors, read_class_names

    model = YoloV3('./smoke_preview.pb')
    classes = read_class_names("./data/my_data/data.names")
    color_table = get_color_table(80)

    #images = []
    files = './459.jpg'
    new_size = [416,416]
    img_ori = cv2.imread(files)
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple(new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    images = img[np.newaxis, :] / 255.
    boxes_, labels_, scores_ = model.run(images)
    boxes_[:, 0] *= (width_ori / float(new_size[0]))
    boxes_[:, 2] *= (width_ori / float(new_size[0]))
    boxes_[:, 1] *= (height_ori / float(new_size[1]))
    boxes_[:, 3] *= (height_ori / float(new_size[1]))

    boxes_[:, 0] = np.ceil(boxes_[:, 0])
    boxes_[:, 2] = np.ceil(boxes_[:, 2])
    boxes_[:, 1] = np.ceil(boxes_[:, 1])
    boxes_[:, 3] = np.ceil(boxes_[:, 3])
    for i in range(len(boxes_)):
        x0, y0, x1, y1 = boxes_[i]
        plot_one_box(img_ori, [x0, y0, x1, y1],
                     label=classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100),
                     color=color_table[labels_[i]])
    # out_name = os.path.join('./data/demo_data/results', 'batch_output_' + os.path.basename(files[idx]))
    out_name = os.path.join('./data/my_data/results', 'batch_output_' + os.path.basename(files))
    cv2.imwrite(out_name, img_ori)
    print('finish!')
