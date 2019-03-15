
1. [关于TensorFlow中的多图(Multiple Graphs)](https://blog.csdn.net/aiya_xiazai/article/details/58701092)
2. [图和会话](https://www.tensorflow.org/programmers_guide/graphs#programming_with_multiple_graphs)
3. [Importing Multiple TensorFlow Models (Graphs)](https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/)


主要是基于[Importing Multiple TensorFlow Models (Graphs)](https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/)的实现，即每个 Graph 建立对应的会话 Session。

代码例子如下所示
```
class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
              # FROM SAVED COLLECTION:            
            self.activation = tf.get_collection('activation')[0]
              # BY NAME:
            self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def run(self, data):
        """ Running the activation operation previously imported """
        return self.sess.run(self.activation, feed_dict={"x:0": data})

model_1 = ImportGraph('models/model_name')
model_2 = ImportGraph('models/different_model')
```
