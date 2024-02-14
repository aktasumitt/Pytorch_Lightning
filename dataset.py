from lightning.pytorch import LightningDataModule
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split


class DatasetLit(LightningDataModule):
    def __init__(self,BATCH_SIZE,DATA_DIR):
        super(DatasetLit,self).__init__()
        self.batch_size=BATCH_SIZE
        self.data_dir=DATA_DIR
    
    def prepare_data(self) :
        
        datasets.FashionMNIST(root=self.data_dir,train=True,download=True)
        datasets.FashionMNIST(root=self.data_dir,train=False,download=True)
        
    def setup(self,stage:str) :
        transformer=transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,),(0.5,))])
        
        if stage=="fit":
            full_dataset=datasets.FashionMNIST(root=self.data_dir,train=True,download=True,transform=transformer)
            self.train_dataset,self.validation_dataset=random_split(full_dataset,lengths=[(len(full_dataset)-10000),10000])
        
        if stage=="test":
            self.test_dataset=datasets.FashionMNIST(root=self.data_dir,train=False,download=True,transform=transformer)
        
        if stage=="predict":
            self.predict_dataset=datasets.FashionMNIST(root=self.data_dir,train=False,download=True,transform=transformer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset,batch_size=self.batch_size,shuffle=False)
    
    def test_dataloader(self) :
        return DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False)
    
    def predict_dataloader(self):
        return DataLoader(self.predict_dataset,batch_size=self.batch_size,shuffle=False)

        