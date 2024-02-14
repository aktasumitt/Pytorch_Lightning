# Pytorch_Lightning

## Introduction:

- In this project, I aimed to train a CNN model with Pytorch Lightning. PyTorch Lightning makes training deep neural networks easier by removing much of the boilerplate code. However, although Lightning's focus is on simplicity and flexibility, it also allows us to use many advanced features, such as multi-GPU support and fast, low-precision training. It is very useful i think.

## Dataset:
- I used the FashionMnist Dataset for this project, which consists of 10 labels with total training 60000 and test 10000 images.
- I randomly split the train dataset into training and validation sets with size of images (50000,10000).
- And I divided them into mini-batches with a batch size of 100. 
- img sizes are (28 x 28 x 1)

## Train:

- I built the model with a custom basic CNN structure.
- I chose Adam optimizer with a learning rate of 0.001 and used CrossEntropyLoss as the loss function. I trained the model for 20 epochs.

## Results:
- After 20 epochs, the model achieved approximately 95.5% accuracy on both the training, validation, and test sets.
- I think training was faster than custom training and it was easier to write.
- We can save checkpoints and use tensorboard simplier than manuel.  

## Usage: 
- You can train the model by setting "TRAIN" to "True" in config file and your checkpoint will save in "config.CHECKPOINTS_PATH"
- You can test the model by settiing "TEST" to "True" in config file.
- Then you can predict the images placed in the Prediction folder by setting the "LOAD_CHECKPOINTS" and "PREDICT" values to "True" in the config file.

### For More Details :  https://lightning.ai



