# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# For Visualizations
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def plot_image(i, predictions_array, true_label, img, class_names):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
  
# Creating a CNN class
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object
        def __init__(self, num_classes):
            super(ConvNeuralNet, self).__init__()
            self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
            self.relu_1 = nn.ReLU()
            self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
            self.relu_2 = nn.ReLU()
            self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

            self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
            self.relu_3 = nn.ReLU()
            self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
            self.relu_4 = nn.ReLU()
            self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

            self.fc1 = nn.Linear(1600, 128)
            self.relu_5 = nn.ReLU()
            self.fc2 = nn.Linear(128, num_classes)

        # Progresses data across layers
        def forward(self, x):
            out = self.conv_layer1(x)
            out = self.relu_1(out)
            out = self.conv_layer2(out)
            out = self.relu_2(out)
            out = self.max_pool1(out)

            out = self.conv_layer3(out)
            out = self.relu_3(out)
            out = self.conv_layer4(out)
            out = self.relu_4(out)
            out = self.max_pool2(out)

            out = out.reshape(out.size(0), -1)

            out = self.fc1(out)
            out = self.relu_5(out)
            out = self.fc2(out)
            return out
  
def cnn(device):
    # Define relevant variables (called hyperparameters) for the ML task
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 50

    # Use transforms.compose method to reformat images for modeling,
    # and save to variable all_transforms for later use
    all_transforms = transforms.Compose([transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                            std=[0.2023, 0.1994, 0.2010])
                                        ])
    # Create Training dataset
    train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                                train = True,
                                                transform = all_transforms,
                                                download = True)

    # Create Testing dataset
    test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                                train = False,
                                                transform = all_transforms,
                                                download=True)

    # Instantiate loader objects to facilitate processing
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)


    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = True)

    
    model = ConvNeuralNet(num_classes)
    model.to(device)

    # Set Loss function with criterion
    criterion = nn.CrossEntropyLoss()

    # Set optimizer with optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

    total_step = len(train_loader)

    # We use the pre-defined number of epochs to determine how many iterations to train the network on
    for epoch in range(num_epochs):
    # Load in the data in batches using the train_loader object
        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Run on validation

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

    from tqdm import tqdm
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the {} test images: {} %'.format(50000, 100 * correct / total))

    # Visualizations
    # The code shown below is taken from TensorFlow.
    # The original code utilized TensorFlow, and our team converted this to be compatible with PyTorch
    # Output figure without noise, and figure with noise for comparison

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Testing Dataset
    image, label = test_dataset[0]
    image = image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(image) # pass test data
        prediction = torch.argmax(logits, dim=1)

    print(f"Predicted Class ID: {prediction.item()}")
    print(f"Actual Class ID: {label}")

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(image, prediction, test_dataset, class_names)
    plt.subplot(1,2,2)
    plot_value_array(image, prediction,  test_dataset)
    plt.show()

    # Training Dataset
    image, label = train_dataset[0]
    image = image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        logits = model(image) # pass test data
        prediction = torch.argmax(logits, dim=1)

    print(f"Predicted Class ID: {prediction.item()}")
    print(f"Actual Class ID: {label}")

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(image, prediction, train_dataset)
    plt.subplot(1,2,2)
    plot_value_array(image, prediction,  train_dataset)
    plt.show()
