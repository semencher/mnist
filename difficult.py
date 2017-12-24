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

# Объявляем две функции, которые будут удобны, для создания матриц весов и смещений (b).
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# Вычисляем свертку для входов x, c ядром W, strides - шаг скольжения окна для каждой размерности входа. padding - тип спользуемого алгоритма заполнения.
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# ksize - размер окна для каждой размерности входного тензора.
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# 5 на 5 - размер патча (ядра, накладки), 1 - количество входных каналов, 32 - количество выходных каналов.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 28 на 28 - размер изображения, 1 - количество цветных каналов.
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Применили свертку, применили пулинг и активационную функцию relu.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 5 на 5 - размер патча (ядра, накладки), 32 - количество входных каналов, 64 - количество выходных каналов.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# Применили свертку, применили пулинг и активационную функцию relu.
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Здесь размер изображения уже уменьшен до 7 на 7, мы добавляем full connected слой.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# Предварительно изменив под себя размерность.
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# И не забываем применить relu.
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Для защиты от переобучения применяется отсев. Это вероятность с которой выходные нейроны сохранятся во время отсева.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Последний full connected слой, получаем выходы сети.
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

# Это функция кросс-энтропии. Как я понял, параметр reduction_indices добавляет второе измерение y то есть все примеры в пакете, после чего reduce_mean
# вычисляет среднее.
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Это более стабильный вариант предыдущего.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# Обучение. Алгоритм обратного распространения ошибки. 0,01 - коэфициент алгоритма градиентного спуска для обучения. cross_entropy - сводится к 0.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# tf.argmax(y,1) - наша модель дает наиболее вероятный ответ, tf.argmax(y_,1) - правильный ответ, мы сравниваем.
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

# [1,0,1,1] -> 0.75 (reduce_mean - вычисляет среднее, cast - [True, False, True, True] -> [1,0,1,1])
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Обучаем и проверяем.
with tf.Session() as sess:
  # Инициализируем переменные.
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
  	# Берем 50 случайных наборов для обучения.
    batch = mnist.train.next_batch(50)
    # Если текущее i делиться на 100 без остатка, то подсчитываем точность модели в текущем состоянии и выводим.
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    # Проводим обучение.
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  # Берем 5000 случайных тестовых данных и проверяем нашу модель на них, больше тестовых данных не позволяет взять нехватка памяти.
  # keep_prob - параметр для dropout.
  testSet = mnist.test.next_batch(5000)
  print("test accuracy %g"%accuracy.eval(feed_dict={ x: testSet[0], y_: testSet[1], keep_prob: 1.0}))



"""
На 20000 - 99.1%.
"""