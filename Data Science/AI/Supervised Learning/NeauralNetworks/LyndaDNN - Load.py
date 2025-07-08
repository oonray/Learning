import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import minmax_scale

test_data = pd.read_csv("./Ex_Files_TensorFlow/sales_data_test.csv",dtype=float)
X_test , Y_test = minmax_scale(test_data.drop("total_earnings",axis=1), [0,1]), minmax_scale(test_data["total_earnings"],[0,1]).reshape(-1,1)

data_len = 9

with tf.variable_scope("input"):
    X = tf.placeholder(tf.float32, shape=(None, data_len))


def get_model(inn):
    with tf.variable_scope("layer1"):
        l1_w = tf.get_variable(name="l1_weights",shape=(data_len, 100), initializer=tf.contrib.layers.xavier_initializer())
        l1_b = tf.get_variable(name="l1_biases",shape=[100], initializer=tf.zeros_initializer())
        l1_out = tf.nn.relu(tf.matmul(inn,l1_w) + l1_b, name="layer1_relu")

    with tf.variable_scope("layer2"):
        l2_w = tf.get_variable(name="l2_weights",shape=[100, 50], initializer=tf.contrib.layers.xavier_initializer())
        l2_b = tf.get_variable(name="l2_biases",shape=[50], initializer=tf.zeros_initializer())
        l2_out = tf.nn.relu(tf.matmul(l1_out,l2_w) + l2_b, name="layer2_relu")

    with tf.variable_scope("layer3"):
        l3_w = tf.get_variable(name="l3_weights", shape=[50, 100], initializer=tf.contrib.layers.xavier_initializer())
        l3_b = tf.get_variable(name="l3_biases", shape=[100], initializer=tf.zeros_initializer())
        l3_out = tf.nn.relu(tf.matmul(l2_out,l3_w) + l3_b, name="layer3_relu")

    with tf.variable_scope("output"):
        o_w = tf.get_variable(name="o_weights",shape=(100,1),initializer=tf.contrib.layers.xavier_initializer())
        o_b = tf.get_variable(name="o_biases",shape=[1], initializer=tf.zeros_initializer())
        return tf.matmul(l3_out,o_w) + o_b


with tf.variable_scope("predict"):
    model = get_model(X)

with tf.variable_scope("cost"):
    Y = tf.placeholder(tf.float32, shape=(None, 1))
    cost = tf.reduce_mean(tf.squared_difference(model, Y, name="SquaredDifference"), name="Mean")

with tf.variable_scope("train"):
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with tf.variable_scope("log"):
    tf.summary.scalar("cost", cost)
    cost_scalar = tf.summary.merge_all()

with tf.Session() as sess:
    train_logger = tf.summary.FileWriter("./log/train")
    test_logger = tf.summary.FileWriter("./log/test")

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess,"./model/saved")

    ret = sess.run(model, feed_dict={X: X_test})

    print(ret)
