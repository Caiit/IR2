import json
import tensorflow as tf


class ResourcePrediction():
    def __init__(self, model_folder):
        self.load_model(model_folder)

    def load_model(self, model_folder):
        self.params = json.loads(open(model_folder + '/parameters.json').read())
        self.model_file = tf.train.latest_checkpoint(model_folder + "/checkpoints")

    def predict(self, x):
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            sess = tf.Session(config=session_conf)

            # TODO: do this in load instead of every time?
            with sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(self.model_file))
                saver.restore(sess, self.model_file)

                input_x = graph.get_operation_by_name("input_x").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                probs = graph.get_operation_by_name("output/probs").outputs[0]

                prediction = sess.run(probs, {input_x: x, dropout_keep_prob: 1.0})

                return prediction
