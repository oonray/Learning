import gym
import numpy as np
import tensorflow as tf

LR = 1e-3

env = gym.make("CartPole-v1")
env.reset()

goal_steps = 500
score_rec = 50
init_games = 10000
epocs = 5

render=True

x = tf.placeholder("float")
y = tf.placeholder("float")

def run_some_sample_games():
    for ep in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render() if render else ""
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

def Populate():
    T_data = []
    Scores = []
    Aceppted_Scores = []

    for _ in range(init_games):
        scores = 0
        game_memory = []
        prev_observation = []


        for _ in range(goal_steps):
            action = np.random.randint(0,2)
            observation, reward, done, info = env.step(action)


            if len(prev_observation) > 0:
                game_memory.append([prev_observation,action])

            prev_observation = observation
            scores += reward
            if done:
                env.reset()
                break

        if scores >= score_rec:
            Aceppted_Scores.append(scores)
            for data in game_memory:
                if data[1]==1:
                    output = [0,1]
                elif data[1]==0:
                    output = [1,0]


                T_data.append([data[0],output])
        env.reset()
        Scores.append(scores)

    T_data_save =np.array(T_data)
    np.save("./Training.npy",T_data_save)
    return T_data



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



def NN_Train(data):
    model = NN_Model(data)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=y))
    optimiser = tf.train.AdamOptimizer(LR).minimize(cost)

    train_x = [i[0] for i in data_t]
    train_y = [i[1] for i in data_t]

    test_x = [i[0] for i in data_t]
    test_y = [i[1] for i in data_t]

    loss = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epocs):
                    _, c = sess.run([optimiser,cost],feed_dict={x:train_x, y:train_y})
                    loss += c
                    if i % 1 == 0:
                        print("Epoc {} : Loss {}".format(i,loss))

        corr = tf.equal(tf.argmax(model,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(corr,"float"))

        print("Accracy {}".format(accuracy.eval({x:test_x,y:test_y})))

        saver = tf.train.Saver()
        saver.save(sess,"./Model.mod")


data_s = Populate()
data_t = data_s[:int((len(data_s)/3)*2)]
data_te = data_s[:int(-(len(data_s)/3))]
NN_Train(x)



