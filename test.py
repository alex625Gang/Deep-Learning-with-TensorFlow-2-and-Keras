import tensorflow as tf

v1 = tf.constant([1,2,3,4])
v2 = tf.constant([2,1,5,3])
v_add = tf.add(v1, v2)
with tf.Session() as sess:
    print(sess.run(v_add))