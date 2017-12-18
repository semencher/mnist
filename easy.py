# Импортируем tensorflow с псевдонимом tf, начинаем сессию работы с tensorflow.
import tensorflow as tf
sess = tf.InteractiveSession()

# Импортируем модуль input_data, с помощью которого далее берем данные mnist, с условием что все подписи (т.е. цифры) будут представлены в виде one hot векторов.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Значение которое мы будем называть входным, float32 - тип чисел, [None, 784] - двумерный тензор с плавающей точкой 28*28=784, None - количество может быть
# любое.
x = tf.placeholder(tf.float32, shape=[None, 784])

# Это как я понимаю изменяемый тензор, который мы создаем для матрицы весов и вектора смещения, заполняем нулями.
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Наша модель. Это тоже Wx, но так как уже не два D, то правильно делать так.
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Это правильные ответы, вектор для ввода.
y_ = tf.placeholder(tf.float32, [None, 10])

# Для оценки текущего состояния модели будем использовать кросс-энтропию.

# Это функция кросс-энтропии. Как я понял, параметр reduction_indices добавляет второе измерение y то есть все примеры в пакете, после чего reduce_mean
# вычисляет среднее.
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Это более стабильный вариант предыдущего.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Обучение. Алгоритм обратного распространения ошибки. 0,5 - коэфициент алгоритма градиентного спуска для обучения. cross_entropy - сводится к 0.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Запускаем модель!
sess = tf.InteractiveSession()

# Инициализируем переменную.
tf.global_variables_initializer().run()

# Обучаем за 1000 шагов.
for _ in range(1000):
# Получаем 100 случайных данных обучающих данных. (входы, правильные ответы)
  batch_xs, batch_ys = mnist.train.next_batch(100)
# Обучаем, передаем данные и обучалку.
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# tf.argmax(y,1) - наша модель дает наиболее вероятный ответ, tf.argmax(y_,1) - правильный ответ, мы сравниваем.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# [1,0,1,1] -> 0.75
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Наконец запускаем модель на наших тестовых данных.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))