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