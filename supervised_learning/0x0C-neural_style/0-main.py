# # #!/usr/bin/env python3

# # import matplotlib.image as mpimg
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import tensorflow as tf

# # NST = __import__('0-neural_style').NST


# # if __name__ == '__main__':
# #     style_image = mpimg.imread("C:/Users/CAMPUSNA/Desktop/ML Projects/supervised learnning/12.neural Style Transfer/starry_night.jpg")
# #     content_image = mpimg.imread("C:/Users/CAMPUSNA/Desktop/ML Projects/supervised learnning/12.neural Style Transfer/golden_gate.jpg")

# #     print(NST.style_layers)
# #     print(NST.content_layer)
# #     nst = NST(style_image, content_image)
# #     scaled_style = nst.scale_image(style_image)
# #     scaled_content = nst.scale_image(content_image)
# #     print(type(nst.style_image), nst.style_image.shape, np.min(nst.style_image),
# #                np.max(nst.style_image))
# #     print(type(nst.content_image), nst.content_image.shape, np.min(nst.content_image),
# #                np.max(nst.content_image))
# #     print(nst.alpha)
# #     print(nst.beta)
# #     print(tf.executing_eagerly())
# #     assert(np.array_equal(scaled_style, nst.style_image))
# #     assert(np.array_equal(scaled_content, nst.content_image))

# #     plt.imshow(nst.style_image[0])
# #     plt.show()
# #     plt.imshow(nst.content_image[0])
# #     plt.show()
# #!/usr/bin/env python3

# import numpy as np
# import tensorflow as tf
# NST = __import__('0-neural_style').NST

# if __name__ == '__main__':
#       np.random.seed(1)
#       image = np.random.uniform(0, 256, size=(100, 200, 3))
#       scaled = NST.scale_image(image)

#       print(scaled.shape)
#       image = image.transpose((1, 0, 2))
#       scaled = NST.scale_image(image)

#       print(scaled.shape)
#       image = np.random.uniform(0, 256, size=(1000, 2000, 3))
#       scaled = NST.scale_image(image)

#       print(scaled.shape)
#       image = image.transpose((1, 0, 2))
#       scaled = NST.scale_image(image)

#       print(scaled.shape)
#!/usr/bin/env python3

import numpy as np
NST = __import__('0-neural_style').NST

if __name__ == '__main__':
      np.random.seed(11)
      style_image = np.random.uniform(0, 256, size=(1000, 1000, 3))
      content_image = np.random.uniform(0, 256, size=(500, 1000, 3))
      try:
            nst = NST(style_image, content_image, alpha=2)
      except TypeError as e:
            print(str(e))
      try:
            nst = NST(style_image, content_image, alpha=-1)
      except TypeError as e:
            print(str(e))