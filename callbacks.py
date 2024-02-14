from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch import LightningModule, Trainer


def Save_MyCheckpoint(CHECKPOINT_PATH):

    checkpoint = ModelCheckpoint(dirpath=CHECKPOINT_PATH,
                                 filename="{epoch}-{step}",
                                 save_top_k=1,
                                 monitor="val_loss",
                                 mode="min",
                                 save_last=True,
                                 save_on_train_epoch_end=True,
                                 )

    return checkpoint


class MyCallback(Callback):
    def __init__(self, Load_checkpoint, Save_checkpoint=True):

        super(MyCallback, self).__init__()
        self.Load_checkpoint = Load_checkpoint
        self.Save_checkpoint = Save_checkpoint

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):

        if self.Load_checkpoint:
            print("Checkpoint is loading...\nTraininig is starting...")

        else:
            print("Traininig is starting from scratch")

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule):
        print("\nTraining Finished...")

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):

        if self.Save_checkpoint:
            print("\nCheckpoint is saving...")

        else:
            print("")
