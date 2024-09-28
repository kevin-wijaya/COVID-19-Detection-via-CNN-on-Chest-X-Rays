# import libraries
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import multiprocessing, torch, torchvision, gc, random
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from PIL import Image

# set up the seed for the random number generator
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# init constant variable
SHAPE = (728, 728)
DATASET_PATH = 'drive/MyDrive/Colab Notebooks/data/Covid19-dataset'
CORE = multiprocessing.cpu_count() // 2
DISPLAY_ROW = 4
DISPLAY_COL = 8
EPOCHS = 10
BATCH_SIZE = 32
NUM_CLASSES = 3
CLASSES = ['Covid', 'Normal', 'Viral Pneumonia']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define class DatasetHandler
class DatasetHandler:
  transformation = torchvision.transforms.Compose([
        torchvision.transforms.Resize(SHAPE),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

  def load_image(self, file):
    image = Image.open(file.stream)
    image_tensor = self.transformation(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    return torch.autograd.Variable(image_tensor)

  def load_dataset(self):
    train_dataset = torchvision.datasets.ImageFolder(root=f'{DATASET_PATH}/train', transform=self.transformation)
    test_dataset = torchvision.datasets.ImageFolder(root=f'{DATASET_PATH}/test', transform=self.transformation)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=CORE)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=CORE)

    return train_loader, test_loader

  @staticmethod
  def img_show(suptitle, images, label_names, label_ground_truth, label_prediction=None):
    plt.figure(figsize=(19, 11))
    plt.suptitle(suptitle, fontsize=15, fontweight='bold', y=0.92)

    for i in range(BATCH_SIZE):
      image = images[i].numpy().transpose((1, 2, 0))
      image_unnormalized = image * 0.5 + 0.5

      plt.subplot(DISPLAY_ROW, DISPLAY_COL, i+1)
      plt.imshow(image_unnormalized)

      title = label_names[label_ground_truth[i]] + f' [{label_ground_truth[i]}]'
      if label_prediction: title += f'\nPredict: {label_names[label_prediction[i]]}'
      plt.title(title, fontsize=8, pad=10, fontweight='bold')

# create architecture model
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=0)
        self.drop = torch.nn.Dropout2d(p=0.2)
        self.fc = torch.nn.Linear(in_features=178 * 178 * 24, out_features=NUM_CLASSES)

    def forward(self, x):
        x = torch.nn.functional.relu(self.pool(self.conv1(x)))
        x = torch.nn.functional.relu(self.pool(self.conv2(x)))
        x = torch.nn.functional.relu(self.drop(self.conv3(x)))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = x.view(-1, 178 * 178 * 24)
        x = self.fc(x)
        return torch.nn.functional.log_softmax(x, dim=1)
    
# create model builder
class ModelBuilder:
  def __init__(self, model, save_weights=False):
    self.model = model
    self.history = []
    self.save_weights = save_weights
    self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    self.loss_func = torch.nn.CrossEntropyLoss()

  def clean_memory(self):
    torch.cuda.empty_cache()
    gc.collect()

  def fit(self, train_loader, test_loader):
    for epoch in range(EPOCHS):
      print(f'Epoch {epoch+1}/{EPOCHS}')

      total_loss_train = 0
      total_loss_test = 0
      y_pred_train, y_actual_train = [], []
      y_pred_test, y_actual_test = [], []

      format = '{n_fmt}/{total_fmt} [{bar}] {rate_fmt}{postfix}'
      with tqdm(total=len(train_loader), bar_format=format) as progress:

        self.model.train()
        for images, labels in train_loader:
          images, labels = images.to(DEVICE), labels.to(DEVICE)
          self.optimizer.zero_grad(); logits = self.model(images)
          loss = self.loss_func(logits, labels); total_loss_train += loss.item()

          y_pred_train.extend(np.argmax(logits.detach().cpu().tolist(), axis=1).tolist())
          y_actual_train.extend(labels.cpu().tolist())

          loss.backward(); self.optimizer.step(); progress.update(1)

        self.model.eval()
        for images, labels in test_loader:
          images, labels = images.to(DEVICE), labels.to(DEVICE)
          self.optimizer.zero_grad(); logits = self.model(images)
          loss = self.loss_func(logits, labels); total_loss_test += loss.item()

          y_pred_test.extend(np.argmax(logits.detach().cpu().tolist(), axis=1).tolist())
          y_actual_test.extend(labels.cpu().tolist())

        self.history.append({
          'loss':         total_loss_train/len(train_loader),
          'accuracy':     accuracy_score(y_pred_train, y_actual_train),
          'val_loss':     total_loss_test/len(test_loader),
          'val_accuracy': accuracy_score(y_pred_test, y_actual_test)
        })

        progress.set_postfix(self.history[-1], refresh=False)
        self.clean_memory()

  def predict(self, image):
    self.model.eval()
    datasetHandler = DatasetHandler()
    features_image = datasetHandler.load_image(image)
    logits = self.model(features_image.to(DEVICE))
    index = logits.detach().cpu().numpy().argmax()
    distribution = torch.exp(logits).detach().cpu().numpy()[0]
    print(distribution)
    probabilities = {CLASSES[i]: str(distribution[i]) for i in range(len(CLASSES))}
    return CLASSES[index], probabilities

  def save(self, path='model.pt'):
    torch.save(self.model.state_dict(), path)
    print('Saving Model Successfully')

  def load(self, path='model.pt'):
    self.model.load_state_dict(torch.load(path, map_location=DEVICE))