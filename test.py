import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
# Example class vector
y = ['alex','tom','yost','eric']

label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)
print(y_int)
# Number of classes
num_classes = len(label_encoder.classes_)

# Convert class vector to binary class matrix
y_categorical = tf.keras.utils.to_categorical(y_int, num_classes)

print(y_categorical)