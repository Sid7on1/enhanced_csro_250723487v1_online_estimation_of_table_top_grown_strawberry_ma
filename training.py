import os
import logging
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from strawberry_model import StrawberryModel
from cycle_gan import CycleGAN
from yolo_v8_seg import YOLOv8Seg
from utils import load_data, save_model, load_model, get_logger

# Define constants and configuration
CONFIG_FILE = 'config.yaml'
MODEL_DIR = 'models'
DATA_DIR = 'data'
LOG_DIR = 'logs'

# Define constants and configuration
class Config:
    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.model_dir = self.config['model_dir']
        self.data_dir = self.config['data_dir']
        self.log_dir = self.config['log_dir']
        self.batch_size = self.config['batch_size']
        self.epochs = self.config['epochs']
        self.learning_rate = self.config['learning_rate']
        self.model_name = self.config['model_name']

# Define the training pipeline class
class Trainer:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(self.config.log_dir)
        self.model_dir = self.config.model_dir
        self.data_dir = self.config.data_dir
        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs
        self.learning_rate = self.config.learning_rate
        self.model_name = self.config.model_name

    def load_data(self):
        self.logger.info('Loading data...')
        self.train_data, self.val_data = load_data(self.data_dir)

    def create_model(self):
        self.logger.info('Creating model...')
        self.model = StrawberryModel()
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, self.model_name)))
        self.model.eval()

    def train(self):
        self.logger.info('Training model...')
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()
        for epoch in range(self.epochs):
            for batch in self.train_data:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            self.logger.info(f'Epoch {epoch+1}, Loss: {loss.item()}')
        self.logger.info('Training complete.')

    def evaluate(self):
        self.logger.info('Evaluating model...')
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_data:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        self.logger.info(f'Validation Loss: {total_loss / len(self.val_data)}')

    def save_model(self):
        self.logger.info('Saving model...')
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, self.model_name))

# Define the main function
def main():
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--config', default=CONFIG_FILE, help='Configuration file')
    args = parser.parse_args()
    config = Config(args.config)
    trainer = Trainer(config)
    trainer.load_data()
    trainer.create_model()
    trainer.train()
    trainer.evaluate()
    trainer.save_model()

# Define the get_logger function
def get_logger(log_dir):
    logger = logging.getLogger('trainer')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# Define the load_data function
def load_data(data_dir):
    train_data = []
    val_data = []
    for file in os.listdir(data_dir):
        if file.endswith('.jpg'):
            image = Image.open(os.path.join(data_dir, file))
            image = transforms.ToTensor()(image)
            label = np.array([1.0, 2.0, 3.0])  # Replace with actual label
            train_data.append((image, label))
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    return train_data, val_data

# Define the save_model function
def save_model(model_dir, model_name, model):
    torch.save(model.state_dict(), os.path.join(model_dir, model_name))

# Define the load_model function
def load_model(model_dir, model_name):
    model = StrawberryModel()
    model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
    return model

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    main()