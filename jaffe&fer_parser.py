import tensorflow as tf
import numpy as np
import jaffe_parser
import matplotlib.pyplot as plt
import fer_parser
import pickle

"""

JAFFE dataset parsing
"""

"""
parser = jaffe_parser.Jaffee_Parser()

X_data = parser.images_to_tensor()
Y_data = parser.text_to_one_hot()



split_index = int(len(X_data)*.8)

X_tr = X_data[:split_index]
X_te = X_data[split_index:]

Y_tr = Y_data[:split_index]
Y_te = Y_data[split_index:]
"""


"""
FER 2013 dataset parsing
"""
parser = fer_parser.Fer_Parser()
X_tr, Y_tr, X_te, Y_te = parser.parse_all()
X_tr = X_tr
Y_tr = Y_tr



print X_tr.shape, X_te.shape, Y_tr.shape, Y_te.shape

