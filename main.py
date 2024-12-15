import os
import numpy as np
from scipy.io import loadmat

directories = ["data/SEED4/eeg_feature_smooth/{}/".format(i+1) for i in range(3)] 
channel_coords = [
    ['0', '0', 'AF3', 'FP1', 'FPZ', 'FP2', 'AF4', '0', '0'], 
    ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'], 
    ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'], 
    ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'], 
    ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'], 
    ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'], 
    ['0', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '0'], 
    ['0', '0', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '0', '0']
]

channel_list = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 
    'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 
    'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 
    'O1', 'OZ', 'O2', 'CB2'
]
print(len(channel_coords), len(channel_coords[0]))

coord_dict = {}
for n in range(len(channel_list)):
    for i, l in enumerate(channel_coords):
        for j, x in enumerate(l):
            if channel_list[n] == x:
                coord_dict[n] = (i, j)
print(coord_dict)

n = 24
perSample = ['de_movingAve', 'de_LDS', 'psd_movingAve', 'psd_LDS']
array = np.zeros(shape=(len(directories), len(os.listdir(directories[0])), n, 4, 8, 9, 5, 64)) 
for h, dire in enumerate(directories):
    print(dire)
    data = [loadmat(os.path.join(dire, file)) for file in os.listdir(dire)]
    for i, bigsample in enumerate(data):
        for j in range(n):
            for k, key in enumerate(perSample):
                sample = np.transpose(np.array(bigsample[key + str(j+1)]), (0, 2, 1))
                sample = np.pad(sample, [(0, 0), (0, 0), (0, 64 - sample.shape[2])])
                for l, channel in enumerate(sample):
                    array[h][i][j][k][coord_dict[l][0]][coord_dict[l][1]] = channel
print(array.shape)


session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
labels = {0: 'neutral', 1: 'sad', 2: 'fear', 3: 'happy'}

y = np.array(session1_label * 15 + session2_label * 15 + session3_label * 15)

print(y.shape)
y_loso = np.reshape(y, (3,15,24))
print(y_loso.shape)

_X = array.reshape(np.prod(array.shape[0:3]), *array.shape[3:])
print(_X.shape)
X_loso = array[:,:,:,1,:,:,]
X_loso = np.transpose(X_loso, (0,1,2,6,3,4,5))
print(X_loso.shape)

def crossval_loso(generate_model, n_epochs, X, y):
    cvscores = []
    for i in range(10):
        a = [x for x in range(11) if x != i]
        print(a)
        X_train = X[:,a,:,:,:,:,:]
        X_test =  X[:,[i],:,:,:,:,:]
        X_train = X_train.reshape(np.prod(X_train.shape[0:3]), *X_train.shape[3:])
        X_test = X_test.reshape(np.prod(X_test.shape[0:3]), *X_test.shape[3:])
        y_train = y[:,a,:]
        y_test = y[:, [i], :]
        y_train = y_train.flatten()
        y_test = y_test.flatten()
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        model = generate_model()
        model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
        print('------------------------------------------------------------------------')
        print(f'Training for fold {i} ...')
        model.fit(X_train, y_train, epochs=n_epochs, verbose=1) # validation_split=0.2)
        scores = model.evaluate(X_test, y_test, verbose=2)
        print("Score for fold %d - %s: %.6f%%" % (i, model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1])
    print('------------------------------------------------------------------------')
    print("Avg accuracies: %.6f%% (+/- %.6f%%)" % (np.mean(cvscores), np.std(cvscores)))
    
def fully_conv():
    return tf.keras.Sequential([
    tf.keras.layers.SeparableConv1D(
    128, 3, data_format='channels_last', padding="same", activation='relu',input_shape=X.shape[1:]),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.SeparableConv1D(
    128, 3, data_format='channels_last', padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.SeparableConv1D(
    128, 3, data_format='channels_last', padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.SeparableConv1D(
    128, 3, data_format='channels_last', padding="same", activation='relu'),
    tf.keras.layers.MaxPooling1D(
    pool_size=2, strides=2, data_format="channels_last"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

crossval_loso(fully_conv, 100, X_loso, y_loso)



