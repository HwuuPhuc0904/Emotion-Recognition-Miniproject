def fully_conv():
    return tf.keras.Sequential([
    tf.keras.layers.SeparableConv1D(128, 3, data_format='channels_last', padding="same", activation='relu',input_shape=X.shape[1:]),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.SeparableConv1D(128, 3, data_format='channels_last', padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.SeparableConv1D(128, 3, data_format='channels_last', padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.SeparableConv1D(128, 3, data_format='channels_last', padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


