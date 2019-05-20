import tensorflow as tf

class Model:
    def __init__(self):
        self.x = tf.placeholder(tf.int32, name="x")
        self.y = tf.placeholder(tf.int32, name="y")
        self.z = tf.placeholder(tf.int32, name="z")
        self.w = tf.Variable(1, name="w")
        self.k = tf.Variable(2, name="w")
        self.c = tf.constant("Node in g_1")
        self.w = self.x + self.y
        self.k = self.w + self.x - self.z

g_1 = tf.Graph()
with g_1.as_default():
  # Operations created in this scope will be added to `g_1`.
    model = Model()

  # Sessions created in this scope will run operations from `g_1`.
    sess_1 = tf.Session(graph=g_1)


g_2 = tf.Graph()
with g_2.as_default():
  # Operations created in this scope will be added to `g_2`.
    d = tf.constant(20)
    s = tf.placeholder(tf.int32, name="x")
    d = d * s

# `sess_2` will run operations from `g_2`.
    sess_2 = tf.Session(graph=g_2)

# ======== Run Session1 Partially ==========
def init(model):
    with sess_1.as_default():
        with sess_1.graph.as_default():
            sess_1.run(tf.global_variables_initializer())
            handler = sess_1.partial_run_setup([model.w, model.k], [model.x, model.y, model.z])
        return handler

def part_1(handler, model, x, y):
    with sess_1.as_default():
        with sess_1.graph.as_default():
            w = sess_1.partial_run(handler, model.w, {model.x: x, model.y: y})
            print(w)
            return w

# ======== Use Results in Session1 to Run Session 2 =========
def calculate(d, w):
    with sess_2.as_default():
        with sess_2.graph.as_default():
            d_ = sess_2.run(d, feed_dict={s: w})
            print(d_)
            return d_

# ======== Continue Session1 =========
def part_2(handler, model, d_):
    with sess_1.as_default():
        with sess_1.graph.as_default():
            print(sess_1.partial_run(handler, model.k, {model.z: d_}))

handler = init(model)
w = part_1(handler, model, 4, 5)
d_ = calculate(d, w)
part_2(handler, model, d_)