from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate_RMSProp = 0.01
learning_rate_Gradient_Descent = 0.5
training_epochs = 10
softmax_classifier_iterations = 10 #SM iteration 횟수
batch_size = 1000
display_step = 1 # 몇 step 마다 log를 출력할 지 결정한다.
examples_to_show = 10 #reconstruct 된 이미지 총 몇개 보여줄 지 결정
n_hidden_1 = 200 # 첫번째 HL 갯수
n_hidden_2 = 200
n_input = 784 # 이미지의 사이즈, 픽셀 갯수

def build_autoencoder():
    Wh_1 = tf.Variable(tf.random_normal([n_input, n_hidden_1]))
    bh_1 = tf.Variable(tf.random_normal([n_hidden_1]))
    h_1 = tf.nn.sigmoid(tf.matmul(X, Wh_1) + bh_1) # 히든 레이어 1의 activation

    Wh_2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
    bh_2 = tf.Variable(tf.random_normal([n_hidden_2]))
    h_2 = tf.nn.sigmoid(tf.matmul(h_1, Wh_2) + bh_2)

    Wo = tf.Variable(tf.random_normal([n_hidden_2, n_input]))
    bo = tf.Variable(tf.random_normal([n_input]))
    X_reconstructed = tf.nn.sigmoid(tf.matmul(h_2, Wo) + bo)

    return X_reconstructed, h_2

def build_softmax_classfier():
    W = tf.Variable(tf.zeros([n_hidden_2, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_pred = tf.nn.softmax(tf.matmul(extracted_features, W) + b)
    return y_pred

X = tf.placeholder("float", [None, n_input])
y_pred, extracted_features = build_autoencoder()

y_true = X                                   # Output 값 (True Output)을 설정 ( = Input 값 )
y = build_softmax_classfier()                # Predicted Output using Softmax Classifier
y_ = tf.placeholder(tf.float32, [None, 10]) # True Output

reconstruction_cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
initial_optimizer = tf.train.RMSPropOptimizer(learning_rate_RMSProp).minimize(reconstruction_cost)

# 크로스 엔트로피 손실 함수
cross_entripy_cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
softmax_classifier_optimizer = tf.train.GradientDescentOptimizer(learning_rate_Gradient_Descent).minimize(cross_entripy_cost)

finetuning_cost = cross_entripy_cost + reconstruction_cost
finetuning_optimizer = tf.train.GradientDescentOptimizer(learning_rate_Gradient_Descent).minimize(finetuning_cost)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    total_batch = int(mnist.train.num_examples / batch_size)

    for epoch in range(training_epochs):

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_value = sess.run([initial_optimizer, reconstruction_cost], feed_dict={X:batch_xs})

        if epoch % display_step == 0:
            print("Epoch : ", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(cost_value))
    print("Stacked Autoencoder pre-training Optimization Finished!")

    reconstructed_image = sess.run(y_pred, feed_dict={X:mnist.test.images[:examples_to_show]})

    # 원복 이미지와 reconstructed image 를 비교
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(reconstructed_image[i], (28, 28)))
    f.show()
    plt.draw()

    f.savefig("reconstructed_mnist_image.png")   # reconstruction 결과를 png로 저장한다.

    for i in range(softmax_classifier_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(softmax_classifier_optimizer, feed_dict={X:batch_xs, y_:batch_ys})
    print("Softmax Classifier Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Accuracy(before fine-tuning) : ")
    print(sess.run(accuracy, feed_dict={X:mnist.test.images, y_:mnist.test.labels}))

    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, cost_value = sess.run([finetuning_optimizer, finetuning_cost], feed_dict={X:batch_xs, y_:batch_ys})

        if epoch % display_step == 0:
            print("Epoch : ", "%04d" % (epoch + 1), "cost = ", "{:.9f}".format(cost_value))
    print("Fine-tunning softmax model Optimization Finished!")

    print("Accuracy(after fine-tunning) : ")
    print(sess.run(accuracy, feed_dict={X:mnist.test.images, y_:mnist.test.labels}))