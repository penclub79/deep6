import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

learing_rate = 0.001 # 학습률
num_epochs = 30      # 학습횟수
batch_size = 256
display_step = 1
input_size = 784
hidden1_size = 256
hidden2_size = 256
output_size = 10

x = tf.placeholder(tf.float32, shape=[None, input_size])
y = tf.placeholder(tf.float32, shape=[None, output_size])

def build_ANN(x):
    W1 = tf.Variable(tf.random_normal(shape=[input_size, hidden1_size]))
    b1 = tf.Variable(tf.random_normal(shape=[hidden1_size]))
    H1_output = tf.nn.relu(tf.matmul(x, W1) + b1)

    W2 = tf.Variable(tf.random_normal(shape=[hidden1_size, hidden2_size]))
    b2 = tf.Variable(tf.random_normal(shape=[hidden2_size]))
    H2_output = tf.nn.relu(tf.matmul(H1_output, W2) + b2)

    W_output = tf.Variable(tf.random_normal(shape=[hidden2_size, output_size]))
    b_output = tf.Variable(tf.random_normal(shape=[output_size]))
    logits = tf.matmul(H2_output, W_output) + b_output

    return logits

predicted_value = build_ANN(x)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predicted_value, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate=learing_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        average_loss = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, current_loss = sess.run([train_step, loss], feed_dict={x:batch_x, y:batch_y})
            average_loss += current_loss / total_batch

        if epoch % display_step == 0:
            print("반복(Epoch) : %d, 손실 함수(Loss) %f" % ((epoch + 1), average_loss))

        correct_prediction = tf.equal(tf.argmax(predicted_value, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("정확도(Accuracy) : %f" % (accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels})))