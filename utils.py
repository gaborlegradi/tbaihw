import numpy as np
from sklearn.datasets import load_breast_cancer

def load_and_preprocess_data(split_ratio=0.7):
    """
    Loads data from scikit-learn site.
    Data is splitted into train and test set. Function returns with:
     - x_train, y_train, x_test, y_test and
     - mean_x_train, dev_x_train.
    Return values' characteristics at default split_ratio=0.7:
     - x_train.shape: (398, 30), y_train.shape: (398, 1).
     - x_test.shape: (171, 30), y_test.shape: (171, 1).
     - dtype of data matrices: float64.
     - dtype of targets: int32.
     - Meaning of target: 0: Malignant, 1: Benign.
    Since data had possibly been gathered in historical order, it is shuffled with random seed applied.
    Data matrices are also normalized. mean_x_train and dev_x_train contains mean and std dev values 
    for each column in x_train data. This data is needed for preprocessing data for inference and
    postprocessing prediction.
    """
    data_package = load_breast_cancer()
    print(f'Data is loaded and preprocessed.\n',
          f'data_package.data.dtype: {data_package.data.dtype}\n',
          f'data_package.data.shape: {data_package.data.shape}\n',
          f'data_package.target.dtype: {data_package.target.dtype}\n',
          f'data_package.target.shape: {data_package.target.shape}', sep='')
    num_data = data_package.data.shape[0]
    assert num_data == data_package.target.shape[0], f'data.data.shape[0] != data.target.shape[0]'

    # Let train and test data be reproducible.
    np.random.seed(42)
    idx = np.arange(0, num_data)
    # Data seems to be collected in historical order.
    np.random.shuffle(idx)
    np.random.seed(None)

    data = data_package.data[idx]
    target = data_package.target[idx]
    target = np.expand_dims(target, axis=-1)

    split_at = int(num_data * split_ratio)
    assert split_at > 0 and split_at < num_data, f'Improper split_ratio. 0 << split_ratio << 1 is expected.'
    x_train = data[:split_at]
    y_train = target[:split_at]
    x_test = data[split_at:]
    y_test = target[split_at:]
    
    # Maybe it would be better to move it into VAEC class.
    mean_x_train = np.mean(x_train, axis=0)
    dev_x_train = np.std(x_train, axis=0)

    x_train = (x_train - mean_x_train) / dev_x_train
    x_test = (x_test - mean_x_train) / dev_x_train

    print(f'x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}\n',
          f'x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}', sep='')

    return x_train, y_train, x_test, y_test, mean_x_train, dev_x_train