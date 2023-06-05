import zipfile
import numpy as np

train_dataset = {}
for t in range(5):
    zip_file = zipfile.ZipFile('train_dataset_fold{}.zip'.format(t))
    zip_extract = zip_file.extractall()
    train_dataset[t] = zip_extract
    zip_extract.close()

validation_dataset = {}
for t in range(5):
    zip_file = zipfile.ZipFile('validation_dataset_fold{}.zip'.format(t))
    zip_extract = zip_file.extractall()
    validation_dataset[t] = zip_extract
    zip_extract.close()

test_dataset = {}
for t in range(5):
    zip_file = zipfile.ZipFile('test_dataset_fold{}.zip'.format(t))
    zip_extract = zip_file.extractall()
    test_dataset[t] = zip_extract
    zip_extract.close()


np.save('train dataset.npy', train_dataset)
np.save('validation dataset.npy', validation_dataset)
np.save('test dataset.npy', test_dataset)