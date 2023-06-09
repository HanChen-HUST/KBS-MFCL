import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from torch.nn import Sequential, Linear, ReLU
from torchvision import models
from torchvision.models import ResNet50_Weights
import pickle
import json

import json
import os
import re
import pickle
from PIL import Image
from io import BytesIO
from torchvision import transforms



cwd = os.getcwd()
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
bert_config = BertConfig.from_pretrained('bert-base-chinese', output_hidden_states = True)
bert_model = BertModel.from_pretrained('bert-base-chinese', config = bert_config)
resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
for param in resnet50.parameters():
    param.requires_grad = False     

def get_feature_x_origin(sentences):
    MAX_LENGTH = 60
    bert_details = list()
    for sentence in sentences:
        encoded_bert_sent = bert_tokenizer.encode_plus(sentence, max_length=MAX_LENGTH+2, add_special_tokens=True, padding = 'max_length',truncation=True)
        bert_details.append(encoded_bert_sent)
       
    bert_sentence = torch.LongTensor([sample["input_ids"] for sample in bert_details])
    bert_token_type_ids = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
    bert_attention_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
    bert_output = bert_model(input_ids=bert_sentence,attention_mask=bert_attention_mask, token_type_ids=bert_token_type_ids)
    bert_output = bert_output[0]

    root_feature = bert_output[0]
    root_feature = torch.unsqueeze(root_feature, dim=0)
    masked_output = torch.mul(bert_attention_mask.unsqueeze(2), bert_output)
    mask_len = torch.sum(bert_attention_mask, dim=1, keepdim=True)  
    bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
    #output = output.unsqueeze(dim=0) #[1,N-1,27,768]
    return bert_output, root_feature    


def get_feature_visual(image):
    image = torch.unsqueeze(image, dim=0)
    model = list(resnet50.children())[:-2]
    model = torch.nn.Sequential(*model,
            torch.nn.AdaptiveAvgPool2d((4, 4)))
    image = model(image)
    return image

def main(obj):
    filePath = os.path.join(cwd, obj + '.pkl') 
    count =0
    with open(file=filePath, mode='rb') as f:
        try:
            while 1:
                content = pickle.load(f)
                id = np.array(content['id'])
                img = get_feature_visual(content['image'])
                img = img.detach().numpy()
                if len(content['text']) == 1:
                    continue
                feature_x, feature_root = get_feature_x_origin(content['text'])
                feature_x = feature_x.detach().numpy()
                feature_root = feature_root.detach().numpy() 
                edge_list = np.array(content['edge_list'])
                y = np.array(content['label'])
                np.savez(os.path.join(cwd, 'DATA/'+ str(id) + '.npz'), x = feature_x, root_feature = feature_root, img =img, y=y, edge_list=edge_list)
                count = count + 1 
        except Exception:
            print("finish loading ! \n")
    print(f"The number of news items: {count}")



if __name__=='__main__':
    obj ="dataset"
    main(obj)
    print("Finish!")




