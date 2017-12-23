# Импортируем tensorflow с псевдонимом tf, начинаем сессию работы с tensorflow.
import tensorflow as tf
sess = tf.InteractiveSession()

# Импортируем модуль input_data, с помощью которого далее берем данные mnist, с условием что все подписи (т.е. цифры) будут представлены в виде one hot векторов.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Значение которое мы будем называть входным, float32 - тип чисел, [None, 784] - двумерный тензор с плавающей точкой 28*28=784, None - количество может быть
# любое.
x = tf.placeholder(tf.float32, shape=[None, 784])
# Это правильные ответы, вектор для ввода.
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Это как я понимаю изменяемый тензор, который мы создаем для матрицы весов и вектора смещения, заполняем нулями.
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Инициализируем переменные с помощью этой сессии.
sess.run(tf.global_variables_initializer())

# Наша модель.
y = tf.matmul(x,W) + b

# Это функция кросс-энтропии. Как я понял, параметр reduction_indices добавляет второе измерение y то есть все примеры в пакете, после чего reduce_mean
# вычисляет среднее.
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Это более стабильный вариант предыдущего.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Обучение. Алгоритм обратного распространения ошибки. 0,5 - коэфициент алгоритма градиентного спуска для обучения. cross_entropy - сводится к 0.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Обучаем за 1000 шагов.
for _ in range(1000):
  # Получаем 100 случайных данных обучающих данных. (входы, правильные ответы)
  batch = mnist.train.next_batch(100)
  # Обучаем, передаем данные и обучалку.
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

