from config import CONFIG

def loadData():
    with open(CONFIG['dataFile'],'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]