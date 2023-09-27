import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import yaml
import time
import argparse
from sklearn import metrics
from munch import *
from torch_geometric.loader import DataLoader
from propogation import *
from propogation_function import get_augmented_by_drop_edge, get_augmented_by_node_mask,get_augmented_by_node_mask_eignvector,get_augmented_by_drop_edge_eignvector,get_augmented_by_drop_edge_degree,get_augmented_by_node_mask_degree
from model import MFCL
from dataset import *

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="code")
parser.add_argument('--gpu', '-g', type=int, default=0, help='gpu id')
parser.add_argument('--config', '-c', dest='config', help='config file', type=argparse.FileType('r', encoding="utf-8"), required=True)
parser.add_argument('--seed', '-s', type=int,  default=100, help='seed')
parser.add_argument('--alpha', '-a', type=float,  default=0.7, help='alpha')

args = parser.parse_args()
gpu_id = str(args.gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
configs = DefaultMunch.fromDict(yaml.safe_load(args.config.read()))
# set seed
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


torch.cuda.empty_cache()


print('seed: ', seed)
print('batch_size: ', configs.batch_size)
print(torch.cuda.is_available())


alpha = args.alpha
device = torch.device('cuda:'+gpu_id if torch.cuda.is_available() else 'cpu')
loss_func = F.cross_entropy


def evaluate(a, label, max_acc):
    acc = metrics.accuracy_score(label, a)
    precision = metrics.precision_score(label, a, average='weighted')
    recall = metrics.recall_score(label, a, average='weighted')
    F1 = metrics.f1_score(label, a, average='weighted')
    if max_acc < acc:
        print(metrics.classification_report(label, a, digits=4))

    return acc, precision, recall, F1

def trainer(configs, dataname, id_train, id_test):
    unpervised_model = ObjectiveLoss_Graph(num_hidden = 256,num_gc_layers=3).to(device)
    opt = torch.optim.Adam(unpervised_model.parameters(), lr=0.001, weight_decay=1e-4) 
    ITM = ObjectiveLoss_TextImage(configs).to(device)
    optITM = torch.optim.Adam(ITM.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(30):
        unpervised_model.train()
        ITM.train()
        train_set, _ = loadData(dataname, id_train+id_test , id_test)
        train_loader = DataLoader(train_set, batch_size=configs.batch_size, shuffle=True, num_workers = 4)
        loss_all = 0
        for batch_data in train_loader:
            aug1 = get_augmented_by_node_mask(batch_data).to(device)
            aug2 = get_augmented_by_drop_edge(batch_data).to(device) 
            z1, z2 = unpervised_model(aug1,aug2)
            loss = unpervised_model.cal_loss(z1, z2)
            opt.zero_grad() 
            loss_all += loss.item() * (max(batch_data.batch) +1)
            loss.backward()
            opt.step()
            batch_data = batch_data.to(device)
            text, img = ITM(batch_data)
            loss2 = ITM.cal_loss_t(text.squeeze(dim=1), img.squeeze(dim=1))
            optITM.zero_grad()
            loss_all += loss2.item() * (max(batch_data.batch).cpu() +1)
            loss2.backward()
            optITM.step()
        loss = loss_all / len(train_loader)
        print(loss)
    model = MFCL(configs, alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_dataset, test_dataset = loadData(dataname, id_train, id_test)
    train_dataloader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=configs.batch_size)


    best_test_acc = best_precision = best_recall = best_F1 = best_epoch = 0
    count = 0
    for epoch in range(configs.max_epoch):
        model.train()
        unpervised_model.train()
        ITM.train()
        total_loss = 0
        for batch_data in train_dataloader:
            optimizer.zero_grad()
            batch_data = batch_data.to(device)
            graphs_embedding, _ = unpervised_model(batch_data,batch_data)
            text , img = ITM(batch_data)
            y = batch_data.y
            logits = model(text, img, graphs_embedding)
            loss = loss_func(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss
        pred_list = []
        label = []

        model.eval()
        unpervised_model.eval()
        ITM.eval()
        for batch_data in test_dataloader:
            with torch.no_grad():
                batch_data = batch_data.to(device)
                img = batch_data.img
                text = batch_data.root_feature
                y = batch_data.y
                graphs_embedding, _ = unpervised_model(batch_data,batch_data)
                text , img = ITM(batch_data)
                logit = model(text, img, graphs_embedding)
                logit = F.softmax(logit, dim=1)
                pred = torch.argmax(logit, dim=1)            
                pred = pred.cpu().detach().numpy()
                pred_list.append(pred)
                label.append(y.cpu())
        pred_result = np.concatenate(pred_list)
        label_result = np.concatenate(label)
        acc, precision, recall, F1 = evaluate(pred_result, label_result, best_test_acc)
        if best_test_acc < acc:
            count = 0
            best_test_acc = acc
            best_precision = precision
            best_recall = recall
            best_F1 = F1
            best_epoch = epoch
            print("epoch={}\ttotal_loss={}\tmax_acc={}\tacc={}\tprecision={}\trecall={}\tF1={}".format(epoch, total_loss, best_test_acc, acc, best_precision,recall,F1))
        else:
            count += 1
            print("epoch={}\ttotal_loss={}".format(epoch, total_loss))

        if count >= 20:
            break
        
        
    print("Best Accuracy:\tepoch={}\tmax_acc={}\t".format(best_epoch, best_test_acc))
    print("iterator Finish!!\n\n")
    return best_test_acc, best_precision, best_recall, best_F1

def getSet(path):
    content = []
    with open(file=path,mode='r') as f:
        temp = f.readlines()
        for t in temp:
            content.append(t.split('\n')[0])
    return content


dataname = "Chinese"
if __name__ == "__main__":
    t1 = time.time()
    cwd = os.getcwd()
    train = getSet(os.path.join(cwd,"WeiboTrainID.txt"))
    test = getSet(os.path.join(cwd,"WeiboTestID.txt"))
    acc, precision, recall, F1 = trainer(configs, dataname, train, test)
    t2= time.time()
    print("total time:{}".format(t2-t1))
    print("acc={}\tprecision={}\trecall={}\tF1={}".format(acc,precision,recall,F1))



    
    
