#!/usr/bin/env python3

import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf

NST = __import__('2-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("C:/Users/CAMPUSNA/Desktop/ML Projects/supervised learnning/12.neural Style Transfer/starry_night.jpg")
    content_image = mpimg.imread("C:/Users/CAMPUSNA/Desktop/ML Projects/supervised learnning/12.neural Style Transfer/golden_gate.jpg")
    np.random.seed(0)
    nst = NST(style_image, content_image)
    input_layer = tf.constant(np.random.randn(1, 28, 30, 3), dtype=tf.float32)
    gram_matrix = nst.gram_matrix(input_layer)
    print(gram_matrix)
# #### main 0

# # #!/usr/bin/env python3

# import numpy as np
# import tensorflow as tf

# NST = __import__('2-neural_style').NST

# if __name__ == '__main__':
#     np.random.seed(0)
#     tf.enable_eager_execution()
#     input_layer = tf.constant(np.random.randn(1, 320, 280, 3), dtype=tf.float32)
#     gram_matrix = NST.gram_matrix(input_layer)
#     print(gram_matrix)
#     input_layer = tf.Variable(np.random.randn(1, 256, 512, 10), dtype=tf.float32)
#     gram_matrix = NST.gram_matrix(input_layer)
#     print(gram_matrix)