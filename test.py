import tensorflow as tf

shape = [10, 2, 5, 5, 3]
base_initial_state = tf.zeros(shape)
base_initial_state = tf.reduce_sum(base_initial_state, axis=[1, 2, 3])

nb_row = 5
nb_col = 5
stack_size = 10
output_size = nb_row * nb_col * stack_size

reducer = tf.zeros((3, output_size))
initial_state = tf.matmul(base_initial_state, reducer)

output_shp = (-1, nb_col, nb_col, stack_size)
initial_state = tf.reshape(initial_state, output_shp)

sess = tf.Session()
print sess.run(initial_state)
