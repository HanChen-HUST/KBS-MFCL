import torch
from torch.utils.data import Dataset
import numpy as np
import os 
from torch_geometric.data import Data


class Dataset(Dataset):
    def __init__(self, ids, data_path):
        self.ids = ids
        self.data_path = data_path


    def __len__(self):
        return len(self.ids)
    

    def __getitem__(self, index):
        id = self.ids[index]
        data = np.load(os.path.join(self.data_path, id), allow_pickle=True)
        return Data(x=torch.tensor(data['comment'], dtype=torch.float32),
                    root_feature=torch.tensor(data['root_feature'], dtype=torch.float32),
                    img=torch.tensor(data['img'], dtype=torch.float32),
                    y=torch.LongTensor([int(data['y'])]),
                    edge_index=torch.LongTensor(data['edge_list']),
                    id= torch.LongTensor([int(id.split('.')[0])])
                    )
cwd = os.getcwd()

def loadData(dataname, ids_train, ids_test):
    data_path = os.path.join(cwd, dataname)
    print("load the train set")
    train_set = Dataset(ids_train,data_path=data_path)
    print(f"the number of train set: {len(train_set)}")
    print("\n load test set") 
    test_set = Dataset(ids_test,data_path=data_path)
    print(f"the number of test set: {len(test_set)}")
    return train_set, test_set




