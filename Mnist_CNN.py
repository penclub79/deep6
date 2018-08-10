import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

learning_rate = 0.001
num_epochs = 100
batch_size = 10

def build_CNN_classifier(x):

    x_image = tf.reshape(x, [-1, 28, 28, 1]) # gray scale

    W_conv1 = tf.Variable(tf.truncated_normal(shape=[5,5,1,32]))
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)

    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=5e-2))
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    W_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, 1024], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_output = tf.Variable(tf.truncated_normal(shape=[1024, 10], stddev=5e-2))
    b_output = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.matmul(h_fc1, W_output) + b_output

    y_pred = tf.nn.softmax(logits=logits)

    return y_pred, logits

y_pred, logits = build_CNN_classifier(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs):
        batch = mnist.train.next_batch(batch_size)

        if i % batch_size == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1]})
            print("반복(Epoch) : %d, 트레이닝 데이터 정확도 : %f" % (i, train_accuracy))

        sess.run([train_step], feed_dict={x:batch[0], y:batch[1]})
    print("테스트 데이터 정확도 : %f" % accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))
