import os
def changeName(path, cName):
    i = 1
    for filename in os.listdir(path):
        print(path+filename, '=>', path+str(cName)+f'{i:06d}'+'.wav')
        os.rename(path+filename, path+str(cName)+f'{i:06d}'+'.wav')
        i += 1
changeName(os.path.dirname(os.path.abspath(__file__)) + "\\train_data\\",'idx_')