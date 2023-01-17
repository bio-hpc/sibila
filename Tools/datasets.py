import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from os.path import splitext, isfile
FIELD_TARGET = 'class'


def read_data(file_in, io_data=None, head=None):
    if isfile(file_in):
        if io_data:
            io_data.print_m("Read Data {}".format(file_in))
        ext = splitext(file_in)[1]
        if ext == ".pkl":
            if io_data:
                io_data.print_m("End read data (pkl)")
            return pd.read_pickle(file_in)
        elif ext == ".csv":
            if io_data:
                io_data.print_m("End read data (csv)")
            return pd.read_csv(file_in)
        else:
            if io_data:
                io_data.print_e("Unsupported format: " + ext)
    else:
        if io_data:
            io_data.print_e("File not found: " + file_in)


def get_dataset(data_set, io_data=None, predicting=False):
    """
    Load the dataset into memory
    :param data_set
    :return:
    """

    if type(data_set) is str and isfile(data_set):
        if splitext(data_set)[1] == ".csv" or splitext(data_set)[1] == ".pkl":
            dataset = read_data(data_set, io_data)
            features = dataset.columns

            if not predicting:
                y = dataset.iloc[:, -1:]
                last_field = features[len(features) - 1]
            else: # mimic the output class
                y = pd.DataFrame(data=np.zeros((dataset.shape[0], 1)).astype(int))
                last_field = ""

            x = dataset.loc[:, dataset.columns != last_field]
            idx_samples = x.iloc[:, 0].tolist()

            if min(idx_samples) < 0:
                print("\n\nThe first colum of the dataset (ids), cannot contain negative values\n")
                exit()

            x = x.drop(x.columns[0], axis=1)
            y = y[y.columns[0]]

            features = features[1:-1]
            features = pd.DataFrame(features)
            feature_list = features.iloc[:, 0].tolist()  # feature list
            """
            elif splitext(data_set)[1] == ".pkl":
                #exit()
                file_in = data_set
                file_gens = '{}_map.csv'.format(splitext(data_set)[0])
                csv = read_data(file_in, io_data)
                x = csv[0] + csv[2]  # the given datasets are split in 80%:20%
                y = csv[1] + csv[3]
                y = np.array([1 if x == "tumoral" else 0
                              for x in y])  # translate the prediction to: 1 = tummor, 0 = no tumor
    
                features = read_data(file_gens, io_data)  # pd.read_csv(file_gens, header=None)
                feature_list = features.iloc[:, 1].tolist()
            """
        else:
            io_data.print_e("Invalid dataset")
    else:
        io_data.print_e("Dataset not found")

    return np.array(x), np.array(y), feature_list, idx_samples


def split_samples(x, y, train_size, io_data, random_state, idx_samples):
    """
        Split the samples by the percentage indicated in samble_test
    :param x:
    :param y:
    :param train_size:
    :param io_data:
    @param random_state:
    :return:
    """
    # Split the data into training and testing sets

    x = np.insert(x, 0, idx_samples, axis=1)  # add index
    xtr, xts, ytr, yts = train_test_split(x, y, train_size=train_size, random_state=random_state)

    idx_xtr = xtr[:, 0].astype(int)  # get index
    idx_xts = xts[:, 0].astype(int)  # get index

    xtr = np.delete(xtr, 0, axis=1)  # remove index
    xts = np.delete(xts, 0, axis=1)  # remove index

    io_data.print_m('Number of samples: {}'.format(x.shape[0]))
    io_data.print_m('Number of features: {}'.format(x.shape[1]-1))
    io_data.print_m('Target classes: {}'.format(','.join(np.unique(y).astype(str))))

    io_data.print_m('Training:')
    unique, counts = np.unique(ytr, return_counts=True)
    io_data.print_m('\tTotal number of samples: {}'.format(xtr.shape[0]))
    for i, c in enumerate(unique):
        pct = round((counts[i]/xtr.shape[0])*100., 2)
        io_data.print_m('\tNumber of samples of class {}: {} ({}%)'.format(c, counts[i], pct))

    io_data.print_m('Test:')
    unique, counts = np.unique(yts, return_counts=True)
    io_data.print_m('\tTotal number of samples: {}'.format(xts.shape[0]))
    for i, c in enumerate(unique):
        pct = round((counts[i]/xts.shape[0])*100., 2)
        io_data.print_m('\tNumber of samples of class {}: {} ({}%)'.format(c, counts[i], pct))

    return xtr, xts, ytr, yts, idx_xtr, idx_xts
