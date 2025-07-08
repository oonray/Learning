from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
from .Layers.recurrent_h_layer import hidden_layer

class hidden_layer:
    def __init__(self,rnn_size,classes):
        self.n_nodes = rnn_size
        self.data_size = classes

        self.weights = tf.Variable(tf.random_normal([self.n_nodes,self.data_size]))
        self.biases = tf.Variable(tf.random_normal([self.data_size]))

    def nn(self,data):
        return tf.nn.relu(tf.matmul(data, self.weights) + self.biases)

    def output(self,data):
        return tf.matmul(data, self.weights) + self.biases

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)



def rec_nn_model(x):
    layer = hidden_layer(rnn_size,n_classes)

    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x,n_chunks,axis=0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size)
    output, state = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)

    output = layer.output(output[-1])

    return output
