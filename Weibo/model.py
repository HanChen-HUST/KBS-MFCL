import torch
import torch.nn as nn
from propogation import TextImage_Transformer


class MFCL(nn.Module):
    def __init__(self, configs, alpha):
        super(MFCL, self).__init__()
        self.alpha = alpha
        self.contextual_transform = TextImage_Transformer(
            configs.contextual_transform, configs.contextual_transform.output_dim)
        self.contextual_transform2 = TextImage_Transformer(
            configs.contextual_transform, configs.contextual_transform.output_dim)
        self.contextual_transform3 = TextImage_Transformer(
            configs.contextual_transform, configs.contextual_transform.output_dim) 
        self.fc = nn.Sequential(nn.Linear(1536,768),
                                nn.ReLU(True),
                                nn.Linear(768,768))
          
        self.classifier = nn.Sequential(nn.Linear(768*3, 256),
                                        nn.ReLU(True),
                                        nn.BatchNorm1d(256),
                                        nn.Linear(256, 2)
                                        )
        self.init_emb()
    
    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)  

    def forward(self, e, f, c):
        e_length = len(e)
        e_f_mask = torch.ones(e_length, 1).to(e.device)
        f_e_mask = torch.ones(e_length,1).to(e.device)
        c_e_mask = torch.ones(e_length,1).to(e.device)
        text_image = self.fc(torch.cat([e,f],dim=-1)).unsqueeze(dim=1)
    
        e = torch.unsqueeze(e,dim=1)
        f = torch.unsqueeze(f,dim=1)
        c = torch.unsqueeze(c,dim=1)
        e1 = self.contextual_transform(e,e_f_mask,f)
        f1 = self.contextual_transform2(f,f_e_mask,e)
        c1 = self.contextual_transform3(c,c_e_mask,text_image)

        e2 = torch.squeeze(e1,dim=1)
        f2 = torch.squeeze(f1,dim=1)
        c2 = torch.squeeze(c1,dim=1)

        a = self.alpha
        e3 =   (1 - a ) * e2 + a * e.squeeze(dim=1)
        f3 =   (1 - a ) * f2 + a * f.squeeze(dim=1)
        c3 =   (1 - a ) * c2 + a * c.squeeze(dim=1)

        h = torch.cat([e3,f3,c3],dim=-1)
        x = self.classifier(h)
        return x




