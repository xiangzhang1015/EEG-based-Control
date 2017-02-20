import tensorflow as tf
import scipy.io as sc
import numpy as np
import random

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

# the RNN structure is the same with the RNN in train py.
def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # X= tf.transpose(X, [1, 0, 2])
    # X = tf.reshape(X, [-1, n_inputs])


    X_hidd1 = tf.sigmoid(tf.matmul(X, weights['in']) + biases['in'])

    X_hidd2 = tf.matmul(X_hidd1, weights['hidd2']) + biases['hidd2']

    X_hidd3 = tf.matmul(X_hidd2, weights['hidd3']) + biases['hidd3']
    X_hidd3 = tf.reshape(X_hidd3, [-1, batch_size, n_hidden3_units])# (step, batch_size, n_hidden)


    # cell
    ##########################################

    # basic LSTM Cell.
    lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden3_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden3_units, forget_bias=1.0, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # lstm cell is divided into two parts (c_state, h_state)
    ##### TAKE Care, batch_size should be 10 when the testing dataset only has 10 data
    _init_state = lstm_cell.zero_state(n_steps, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_hidd3, initial_state=_init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return (results, X,X_hidd1,X_hidd3, X_hidd3, outputs[-1] )


# input test data
feature = sc.loadmat("/home/xiangzhang/matlabwork/eegmmidb/S1_nolabel6.mat")
all = feature['S1_nolabel6']

np.random.shuffle(all)# mix eeg_all

# this code use 7000 samples as the test data
all=all[0:7000]

feature_all =all[:,0:64]

# feature_all =feature_all.reshape([28000,1,64])
label_all=one_hot(all[:,64])

feature_testing =feature_all

label_testing =label_all

b=feature_testing
#batch split
nodes=64
lameda=0.004
lr=0.005

batch_size=len(all)
# hyperparameters

n_inputs = 64  # MNIST data input (img shape: 11*99)
n_steps = 1 # time steps
n_hidden1_units = nodes   # neurons in hidden layer
n_hidden2_units = nodes
n_hidden3_units = nodes
n_hidden4_units=nodes
n_classes = 6      # MNIST classes (0-9 digits)

# tf Graph input
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, [None, n_inputs],name="features")
    y = tf.placeholder(tf.float32, [None, n_classes],name="label")

    # Define weights
    with tf.name_scope('weights'):
        weights = {
            # (28, 128)
            'in': tf.Variable(tf.random_normal([n_inputs, n_hidden1_units]), name="weights_in"),
            #(128,128)
            'hidd2': tf.Variable(tf.random_normal([n_hidden1_units, n_hidden2_units]), name="weights_hidd2"),
            'hidd3': tf.Variable(tf.random_normal([n_hidden2_units, n_hidden3_units]), name="weights_hidd3"),
            # (128, 10)
            'out': tf.Variable(tf.random_normal([n_hidden3_units, n_classes]), name="weights_out"),

        }
        #these is for thetensorboard, it won't influence the code's work.
        layer_name='layer'
        tf.histogram_summary(layer_name + '/weights', weights)
    with tf.name_scope('biases'):
        biases = {
            # (128, )
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden1_units]), name="biases_in"),
            #(128,)
            'hidd2': tf.Variable(tf.constant(0.1, shape=[n_hidden2_units ]), name="biases_hidd2"),
            'hidd3': tf.Variable(tf.constant(0.1, shape=[n_hidden3_units]), name="biases_hidd3"),
            # (10, )
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes ]), name="biases_out")
        }
        tf.histogram_summary("layer" + 'biases', biases)


pred, layer1,layer2, layer3,layer4,layer5 = RNN(x, weights, biases)
lamena =0.004
l2 = lamena * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())  # L2 loss prevents this overkill neural network to overfit the data
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) + l2  # Softmax loss
tf.scalar_summary('loss', cost)

lr=0.005
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
pred_result =tf.argmax(pred, 1)
label_true =tf.argmax(y, 1)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver=tf.train.Saver()

#For the visualization, every layer's data are tracked.
f1=open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/layer1.csv','wb')
f2=open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/layer2.csv','wb')
f3=open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/layer3.csv','wb')
f4=open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/layer4.csv','wb')
f5=open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/layer5.csv','wb')
f6=open('/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/layer6.csv','wb')

with tf.Session() as sess:
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("logs/", sess.graph)
    #read the model
    saver.restore(sess,"/home/xiangzhang/PycharmProjects/untitled/activity_recognition_practice/eegmmidbmodel/eeg_rawdata_runn_model00.00400.005643.ckpt")# attention: this model should change with different person
    print("The lamda is :",lamena,", Learning rate:",lr,", The accuracy is: ",sess.run(accuracy, feed_dict={
                x: b,
                y: label_testing,
            }))
    np.set_printoptions(threshold='nan') # output all the values, without the apostroph
    # print label_testing
    print ("The predict result",sess.run(pred_result,feed_dict={x:b,y:label_testing}))

    # track all the 6 layers' output, for the visualization
    f1.write(str(sess.run(layer1, feed_dict={x: b, y: label_testing})))
    f2.write(str(sess.run(layer2, feed_dict={x: b, y: label_testing})))
    f3.write(str(sess.run(layer3, feed_dict={x: b, y: label_testing})))
    f4.write(str(sess.run(layer4, feed_dict={x: b, y: label_testing})))
    f5.write(str(sess.run(layer5, feed_dict={x: b, y: label_testing})))
    f6.write(str(sess.run(pred, feed_dict={x: b, y: label_testing})))









