import os

def walkdir(folder):
    for dirpath, _, files in os.walk(folder):
        for filename in files:
            yield (dirpath, filename)


def prepare_datasets(folder):
    X_train=[]
    y_train=[]
    X_test=[]
    y_test=[]
    for dirpath, filename in walkdir(folder):
        folder_train = [folder+'train/pos', folder+'train/neg']
        folder_test = [folder+'test/pos', folder+'test/neg']
        path_file = os.path.join(dirpath, filename)
        target = 1 if 'pos' in dirpath else 0
        if dirpath in folder_train:
            y_train.append(target)
            with open(path_file) as file:
                X_train.append(file.read())
        if dirpath in folder_test:
            y_test.append(target)
            with open(path_file) as file:
                X_test.append(file.read())            
    return X_train, y_train, X_test, y_test