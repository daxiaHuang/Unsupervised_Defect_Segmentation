from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import optimizers
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from SSIM_PIL import compare_ssim as ssim
from keras import backend as K

def autoencoderModel(input_shape):

    input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

    # Encode-----------------------------------------------------------
    x = Conv2D(32, (4, 4), strides=2 , activation='relu', padding='same')(input_img)
    x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    encoded = Conv2D(1, (8, 8), strides=1, padding='same')(x)

    # Decode---------------------------------------------------------------------
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(encoded)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((4, 4))(x)
    x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (8, 8), activation='sigmoid', padding='same')(x)
    # ---------------------------------------------------------------------
    model = Model(input_img, decoded)
    return model

# Own Your Image Directory
# img_dir = ("./Samples/")
# img_files = glob.glob(img_dir + "*.jpeg")
img_dir = ("./textures/texture_1/train/good/")

img_files = glob.glob(img_dir + "*.png")
# Setting Image Propertie
width = 128
height = 128
pixels = width * height * 1  # gray scale

# Load Image
# AutoEncoder does not have to label data
x = []

for i, f in enumerate(img_files):
    img = Image.open(f)
    #     img = img.convert("RGB")
    img = img.convert("L")  # gray sclae
    img = img.resize((width, height), 1)
    data = np.asarray(img)
    x.append(data)
    if i % 10 == 0:
        print(i, "\n", data)

x = np.array(x)
(x_train, x_test) = train_test_split(x, shuffle=False, train_size=0.8, random_state=1)

img_list = (x_train, x_test)
np.save("./obj.npy", img_list)
print("OK", len(x))

# change to float32
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 128, 128, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 128, 128, 1))  # adapt this if using `channels_first` image data format
print (x_train.shape)
print (x_test.shape)

input_shape = x_train.shape[1:]
autoencoder = autoencoderModel(input_shape)
adam = optimizers.Adam(lr=0.0002, decay=0.00001)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()

autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
autoencoder.save_weights('./model.hdf5')
# autoencoder.load_weights('./model.hdf5')
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



img_dir = ("./textures/texture_1/test/defective/")
img_files = glob.glob(img_dir + "*.png")
# Setting Image Propertie
width = 128
height = 128
pixels = width * height * 1  # gray scale

# Load Image
# AutoEncoder does not have to label data
x = []

for i, f in enumerate(img_files):
    img = Image.open(f)
    #     img = img.convert("RGB")
    img = img.convert("L")  # gray sclae
    img = img.resize((width, height), 1)
    data = np.asarray(img)
    x.append(data)
    if i % 10 == 0:
        print(i, "\n", data)

x = np.array(x)
(x_train, x_test) = train_test_split(x, shuffle=False, train_size=0.8, random_state=1)

img_list = (x_train, x_test)
np.save("./obj.npy", img_list)
print("OK", len(x))

# change to float32
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 128, 128, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 128, 128, 1))  # adapt this if using `channels_first` image data format
print (x_train.shape)
print (x_test.shape)
decoded_imgs = autoencoder.predict(x_train)

n = 8  # how many digits we will display
plt.figure(figsize=(20, 5), dpi=100)
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(False)

    # SSIM Encode
    ax.set_title("Encode_Image")

    npImg = x_train[i]
    npImg = npImg.reshape((128, 128))
    formatted = (npImg * 255 / np.max(npImg)).astype('uint8')
    img = Image.fromarray(formatted)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(False)

    # SSIM Decoded
    npDecoded = decoded_imgs[i]
    npDecoded = npDecoded.reshape((128, 128))
    formatted2 = (npDecoded * 255 / np.max(npDecoded)).astype('uint8')
    decoded = Image.fromarray(formatted2)



    value = ssim(img, decoded)

    label = 'SSIM: {:.3f}'

    ax.set_title("Decoded_Image")
    ax.set_xlabel(label.format(value))

plt.show()








