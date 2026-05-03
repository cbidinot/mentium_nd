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
  
def cnn(device, test_loader, model, test_dataset, train_dataset, class_names):

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

    # Visualizations
    # The code shown below is taken from TensorFlow.
    # The original code utilized TensorFlow, and our team converted this to be compatible with PyTorch
    # Output figure without noise, and figure with noise for comparison

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
