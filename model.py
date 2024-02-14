import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torchmetrics.classification import Accuracy


class ModelLit(LightningModule):
    def __init__(self,num_classes,learning_rate) :
        super(ModelLit,self).__init__()
        self.pred_list=[]
        self.real_labels=[]
        self.LR=learning_rate
        self.accuracy=Accuracy(task="multiclass",num_classes=num_classes)
        
        self.conv=nn.Sequential(torch.nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1,padding_mode="reflect"),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2,stride=2),
                                
                                nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,padding=1,padding_mode="reflect"),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.linear=nn.Sequential(nn.Linear(in_features=(256*7*7),out_features=512),
                                  nn.ReLU(),
                                  nn.Linear(512,128),
                                  nn.ReLU(),
                                  torch.nn.Linear(in_features=128,out_features=num_classes))
                                
                                
    def training_step(self,data):
        image,label = data
        out=self.forward(image)
        loss=nn.CrossEntropyLoss()(out,label)
        acc=self.accuracy(out,label)
            
        self.log_dict({"train_loss":loss, "train_acc":acc},prog_bar=True,on_epoch=True,on_step=False)        
        
        return loss
    
    
    def validation_step(self,data) :
        image,label = data
        out=self.forward(image)
        loss=nn.CrossEntropyLoss()(out,label)
        acc=self.accuracy(out,label)
            
        self.log_dict({"val_loss":loss, "val_acc":acc},prog_bar=True,on_epoch=True,on_step=False)
        
        return loss
    
    def test_step(self,data):
        image,label = data
        out=self.forward(image)
        loss=nn.CrossEntropyLoss()(out,label)
        acc=self.accuracy(out,label)
            
        self.log_dict({"test_loss":loss, "test_acc":acc},prog_bar=True,on_epoch=True,on_step=False)
        
        return loss
    
    def predict_step(self,data):
        image,label = data
        out=self.forward(image)
        _,pred=torch.max(out,1)
        
        self.real_labels.append(label)
        self.pred_list.append(pred)
        
        return {"real_labels":self.real_labels,"pred_labels":self.pred_list}
        

    def configure_optimizers(self):
        return torch.optim.SGD(params=self.parameters(),lr=self.LR)
        
    def forward(self,data):
        
        x=self.conv(data)
        x=x.view(-1,256*7*7)
        out=self.linear(x)
        
        return out

    