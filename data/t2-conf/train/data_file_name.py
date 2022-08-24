import os
def changeName(path, cName):
    for filename in os.listdir(path):
        if not filename.endswith(".wav"):
            continue
        i = int(filename.split('-')[1])-6000
        print(path+filename, '=>', path+str(cName)+f'{i:06d}'+'.wav')
        os.rename(path+filename, path+str(cName)+f'{i:06d}'+'.wav')
changeName(os.path.dirname(os.path.abspath(__file__)) + "\\train_data\\",'idx_')