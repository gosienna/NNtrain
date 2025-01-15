from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Callable
class CreateTorchLoader:
    def __init__(self, 
                 data_reader, 
                 label_df,
                 augmentors: List[Callable] = None,
                 args: dict = None):
        self.data_reader = data_reader
        self.label_df = label_df
        self.args = args
        self.dataset = ImgDataset(self.data_reader, 
                                  self.label_df, 
                                  augmentors,
                                  self.args)
    def create_torch_loader(self):
        return DataLoader(self.dataset, batch_size=self.args['batch_size'], shuffle=self.args['shuffle_data'])    
    
class ImgDataset(Dataset):
    """
    Custom Pytorch Dataset for merging image data and label data for pytorch dataloader
    """
    # data loading
    def __init__(self,
                 data_reader,
                 label_df,
                 augmentors: List[Callable],
                 args: dict):
        self.data_reader = data_reader
        self.label_df = label_df
        # Extract IDs and labels from the DataFrame
        self.ids = label_df['id'].tolist()
        self.labels = label_df[args['label_name']].tolist()
        self.augmentors = augmentors
        self.args = args
    # working for indexing
    def __getitem__(self, index):
         # Get image data using data_reader
        image_data = self.data_reader.get(self.ids[index]).astype(np.float32)
        if self.augmentors is not None:
            for augmentor in self.augmentors:
                image_data = augmentor(image_data)
        # Get corresponding label
        label = self.labels[index]
        # Return both image data and label
        return image_data, label
    # return the length of our dataset
    def __len__(self):
        return len(self.ids)