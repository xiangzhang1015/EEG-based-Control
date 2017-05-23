import tensorflow as tf
import scipy.io as sc
import numpy as np
import random
import pandas as pd
import time

# this function is used to transfer one column label to one hot label
def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]


#EEG eegmmidb person dependent raw data mixed read
#/home/xiangzhang/matlabwork/eegmmidb/
feature = sc.loadmat("/home/xiangzhang/scratch/eegmmidb/S1_nolabel6.mat")
all = feature['S1_nolabel6']

print all.shape

np.random.shuffle(all)  # mix eeg_all
all=all[0:28000]
feature_all =all[:,0:64]
#This command is to transpose the data to 3-D, if your sample is 2-D such as picture, you will need this.
feature_all =feature_all.reshape([28000,1,64])
label_all=one_hot(all[:,64])

# divided data into two parts: training data and testint data,
#this code use 21000 as training data and 7000 as testing data, the batch_size is 3.
middle_number=21000
feature_training =feature_all[0:middle_number]
feature_testing =feature_all[middle_number:28000]
label_training =label_all[0:middle_number]
label_testing =label_all[middle_number:28000]

print(feature_training.shape)
print(feature_testing.shape)

#batch split

a=feature_training
b=feature_testing
nodes=64
lameda=0.004
lr=0.005

batch_size=7000
train_fea=[]
n_group=3
for i in range(n_group):
    f =a[(0+batch_size*i):(batch_size+batch_size*i)]
    train_fea.append(f)
print (train_fea[0].shape)

train_label=[]
for i in range(n_group):
    f =label_training[(0+batch_size*i):(batch_size+batch_size*i), :]
    train_label.append(f)
print (train_label[0].shape)


# hyperparameters
n_inputs = 64  # MNIST data input (img shape: 11*99)
n_steps = 1 # time steps
n_hidden1_units = nodes   # neurons in hidden layer
n_hidden2_units = nodes
n_hidden3_units = nodes
n_hidden4_units=nodes
n_classes = 6      # MNIST classes (0-9 digits)

# tf Graph input

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# x = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="features")
y = tf.placeholder(tf.float32, [None, n_classes])

# Define weights

weights = {
# (28, 128)
'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), trainable=True),
'a': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden1_units]), trainable=True),
#(128,128)
'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units])),
'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units])),
'hidd4': tf.Variable(tf.random_normal([n_hidden3_units, n_hidden4_units])),
# (128, 10)
'out': tf.Variable(tf.random_normal([n_hidden4_units, n_classes]), trainable=True),
}

biases = {
# (128, )
'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units])),
#(128,)
'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units ])),
'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units])),
'hidd4': tf.Variable(tf.constant(0.1, shape=[n_hidden4_units])),
# (10, )
'out': tf.Variable(tf.constant(0.1, shape=[n_classes ]), trainable=True)
}


def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_hidd1 = tf.matmul(X, weights['in']) + biases['in']
    X_hidd2 = tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2']
    X_hidd3 = tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3']
    X_hidd4 = tf.matmul(X_hidd3, weights['hidd4']) + biases['hidd4']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_hidd4, [-1, n_steps, n_hidden4_units])


    # X_hidd1 = tf.sigmoid(tf.matmul(X, weights['in']) + biases['in'])##to make the difference bigger, I add sigmoid function
    #
    # X_hidd2 = tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2']
    #
    # X_hidd3 = tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3']
    #
    # X_hidd4 = tf.reshape(X_hidd3, [-1, batch_size, n_hidden3_units])#

    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden4_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results

pred = RNN(x, weights, biases)
# print(pred)


lamena =lameda
l2 = lamena * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))+l2  # Softmax loss

# tf.scalar_summary('loss', cost)

lr=lr
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    # train_op = tf.train.AdagradOptimizer(l).minimize(cost)# different optimizers
    # train_op = tf.train.RMSPropOptimizer(0.00001).minimize(cost)
    # train_op = tf.train.AdagradDAOptimizer(0.01).minimize(cost)
    # train_op = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
pred_result =tf.argmax(pred, 1)
label_true =tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # merged = tf.merge_all_summaries()
    # writer = tf.train.SummaryWriter("logs/", sess.graph)
    sess.run(init)
    saver = tf.train.Saver()
    step = 0
    start = time.clock()
    # track the accuracy in the training progress
    # filename = "/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/acc/1.text"
    # f = open(filename, 'a')
    while step < 3500:# 1500 iterations
        for i in range(n_group):
            sess.run(train_op, feed_dict={
                x: train_fea[i],
                y: train_label[i],
                })
        if sess.run(accuracy, feed_dict={x: b,y: label_testing,})>0.99:
            print(
            "The lamda is :", lamena, ", Learning rate:", lr, ", The step is:", step, ", The accuracy is: ",
            sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            }))
            break
        if step % 10 == 0:
            hh=sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            })
            print("The lamda is :",lamena,", Learning rate:",lr,", The step is:",step,", The accuracy is:", hh)

            # f.write(str(hh) + '\n')
            # np.set_printoptions(threshold='nan') # output all the values, without the apostrophe

            print("The cost is :",sess.run(cost, feed_dict={
                x: b,
                y: label_testing,
            }))
        step += 1
    endtime=time.clock()

    print "run time:", endtime-start
    #save the model
    # save_path = saver.save(sess, "generalmodel2/eeg_rawdata_runn_model" + str(lamena) + str(
    #     lr) +str(nodes)+str(n_group)+ ".ckpt")
    # print("save to path", save_path)




