import numpy as np
import pickle


from config import CONFIG
from Network import one_hot, Network


# load data
with open(CONFIG['dataFile'],'rb') as f:
    mnist = pickle.load(f)

data = dict()
data['training_images'] = mnist['training_images'].reshape(60000, 28*28)
data['training_images'] = data['training_images'] #- np.mean(data['training_images'], axis=0)
data['training_labels'] = one_hot(mnist['training_labels'], 10)

data['test_images'] = mnist['test_images'].reshape(10000, 28*28) 
mnist['test_images'] = mnist['test_images'] #- np.mean(mnist['test_images'], axis=0)
data['test_labels'] = one_hot(mnist['test_labels'], 10)


def main_keras():
    # import modules
    from keras.layers import Input, Dense, Activation
    from keras.models import Model
    from keras.callbacks import ModelCheckpoint, TensorBoard
    from keras.optimizers import SGD

    # create network
    img_input = Input(shape=(784,), name='input')
    x = Dense(30, name='z0')(img_input)
    x = Activation('sigmoid', name='a0')(x)
    x = Dense(10, name='z1')(x)
    x = Activation('sigmoid', name='a1')(x)
    model = Model(inputs=[img_input], outputs=[x])

    # train network
    checkpoint = ModelCheckpoint('./checkpoints/'+'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min', period=1)
    tb = TensorBoard(log_dir='tbLog', histogram_freq=0, write_images=False, write_graph=False)
    # callbacks_list = [checkpoint, tb]
    callbacks_list = []
    sgd = SGD(lr=CONFIG['learning_rate'], clipvalue=0.5)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
    model.fit(x=data['training_images'], y=data['training_labels'], batch_size=CONFIG['batch_size'], epochs=CONFIG['epochs'], validation_data=(data['test_images'], data['test_labels']), callbacks=callbacks_list)


def main_mine():
    # create network
    model = Network([784, 30, 10])    

    # train network
    model.train(data)


if __name__=='__main__':
    main_mine()




