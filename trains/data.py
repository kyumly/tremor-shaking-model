import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np

torch.random.manual_seed(42)

class CustomDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]




def get_data(data):
    label = data['Result']
    data = data.drop(['mY', 'mZ', "Result"], axis=1)

    st = StandardScaler()
    st_data = st.fit(data).transform(data)

    X_tarin, X_valid, y_train, y_valid= train_test_split(st_data, label,test_size=0.2, random_state=42)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)

    return X_tarin, X_valid, y_train, y_valid

