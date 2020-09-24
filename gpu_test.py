

#python --version

'''
pip install --upgrade tf_slim
pip install tensorflow==2.1
conda install -c conda-forge opencv
vào tils.py để add thư viện còn thiếu
'''



import tensorflow.compat.v1 as tf
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

