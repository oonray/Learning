import tensorflow as tf
import gym
import numpy as np

tf.reset_default_graph()

env = gym.make("CartPole-v1")
env.reset()

goal_steps = 500
score_rec = 50
init_games = 10000
games = 10


x = tf.placeholder("float")
y = tf.placeholder("float")

def NN_Model(data):
    I_W = tf.Variable(tf.random_normal([4,1000]))
    I_B = tf.Variable(tf.random_normal([1000]))

    I_O = tf.nn.relu((tf.matmul(data,I_W) + I_B))

    L1_W = tf.Variable(tf.random_normal([1000,500]))
    L1_B = tf.Variable(tf.random_normal([500]))

    L1_O = tf.nn.relu(tf.matmul(I_O,L1_W) + L1_B)

    O_W = tf.Variable(tf.random_normal([500, 2]))
    O_B = tf.Variable(tf.random_normal([2]))

    return tf.matmul(L1_O,O_W)+ O_B



with tf.Session() as sess:

    model = NN_Model(x)
    pred = tf.argmax(model, 1)
    corr = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(corr, "float"))

    saver = tf.train.Saver()
    saver.restore(sess, "./Model.mod")
    score = 0

    env.reset()
    for i in range(games):
        prev_obs = []
        memory = []

        env.reset()
        for _ in range(goal_steps):
            env.render()
            if len(prev_obs) == 0:
                action = [env.action_space.sample()]
            else:
                action = sess.run(pred, feed_dict={x: [prev_obs]})

            observation, reward, done, info = env.step(action[0])

            score += reward
            prev_obs = observation
            memory.append([observation, action])

            if done:break

    print(score)




