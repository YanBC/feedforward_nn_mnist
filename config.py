CONFIG = dict()

# data 
CONFIG['dataFile'] = './data/mnist.pkl'
CONFIG['imgShape'] = (28, 28)
CONFIG['labelShape'] = (1,)
CONFIG['train_set_size'] = 60000
CONFIG['val_set_size'] = 10000

# train
CONFIG['epochs'] = 30
CONFIG['batch_size'] = 10
CONFIG['shuffle'] = True
CONFIG['shuffle_seed_range'] = 999999
CONFIG['learning_rate'] = 0.01