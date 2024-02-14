import callbacks,dataset,model,config
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

# Ignore Warnings
import warnings
warnings.filterwarnings('ignore')


# Create Linghtenning Data Module
datamodule=dataset.DatasetLit(BATCH_SIZE=config.BATCH_SIZE,DATA_DIR=config.DATA_PATH)


# Create Model and Load Checkpoints In This Model If You Have
if config.LOAD_CHECKPOINTS==True:
    Model = model.ModelLit.load_from_checkpoint(checkpoint_path=config.CHECKPOINT_PATH + "\\last-v2.ckpt", num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE)

else:    
    Model=model.ModelLit(num_classes=config.NUM_CLASSES,learning_rate=config.LEARNING_RATE)


# Create Logger for Tensorboar to Visualize Logg
Logger=TensorBoardLogger(save_dir=config.LOG_PATH,name="lightening_logs")


# Create Lightening Trainer and Add Callbacks
trainer=Trainer(max_epochs=config.MAX_EPOCH,
                devices=1,
                accelerator="gpu",
                callbacks=[callbacks.MyCallback(config.LOAD_CHECKPOINTS,config.SAVE_CHECKPOINTS),
                           callbacks.Save_MyCheckpoint(CHECKPOINT_PATH=config.CHECKPOINT_PATH)],
                logger=Logger)


# Fit Model
if config.TRAIN ==True:
    trainer.fit(model=Model,datamodule=datamodule)


# Test Model
if config.TEST ==True:
    trainer.test(model=Model,datamodule=datamodule,ckpt_path=config.CHECKPOINT_PATH+"\\last.ckpt")


# Prediction_Results
if config.PREDICT==True:   
    pred_results=trainer.predict(model=Model,datamodule=datamodule)