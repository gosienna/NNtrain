from typing import Callable
import os
import pyarrow.parquet as pq
from PickleIO import save_pkl, load_pkl
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
from PIL import Image
#modules to read data from local folder
class Reader:
    def __init__(self, 
                 DATA_PATH: str,
                 in_Memory: bool=False):
        self.DATA_PATH = DATA_PATH
        self.in_Memory = in_Memory
        
    def ids(self,):
        ids = os.listdir(self.DATA_PATH)
        ids = [int(x.split('.')[0]) for x in ids]
        return ids
    
    @abstractmethod
    def get(self, id):
        pass
    
    @abstractmethod
    def __get(self, id):
        pass
    
    @abstractmethod
    def save_local(self, target_folder):
        pass

class PickleReader(Reader):
    #load data directly from local pikle file
    def __init__(self, 
                 DATA_PATH: str,
                 in_Memory: bool=False,
                 formators: list[Callable] = None):
        super().__init__(
            DATA_PATH = DATA_PATH,
            in_Memory = in_Memory
        )
        self.formators = formators
        if in_Memory:
            self.load_data()
    
    def save_local(self, target_folder):
        for id in tqdm(self.ids()):
            data = self.get(id)
            file_path = f"{target_folder}/{id}.pkl"
            if os.path.exists(file_path):
                print(f"{id}.pkl file already exist!")
            else:
                save_pkl(data, file_path)
                
    def get(self, id):
        if self.in_Memory:
            return self.data[id]
        else:
            print('get')
            return self.__get(id)

    def load_data(self, ids = None):
        self.data = {}
        if ids != None:
            files = ids
        else:
            files = self.ids()
        for id in tqdm(files):
            self.data[id] = self.__get(id)
            
    def __get(self, id):
        data = load_pkl(f'{self.DATA_PATH}/{id}.pkl')
        if self.formators is not None:
            for formator in self.formators:
                data = formator(data)
        return data

class ParqReader(Reader):
    #load data directly from local pikle file
    def __init__(self, 
                 DATA_PATH: str,
                 in_Memory: bool=False,
                 formators: list[Callable] = None):
        super().__init__(
            DATA_PATH = DATA_PATH,
            in_Memory = in_Memory
        )
        self.formators = formators
        if in_Memory:
            self.load_data()
    
    def save_local(self, target_folder):
        for id in tqdm(self.ids()):
            data = self.get(id)
            file_path = f"{target_folder}/{id}.pkl"
            if os.path.exists(file_path):
                print(f"{id}.pkl file already exist!")
            else:
                save_pkl(data, file_path)
                
    def get(self, id):
        if self.in_Memory:
            return self.data[id]
        else:
            return self.__get(id)

    def load_data(self, ids = None):
        self.data = {}
        if ids != None:
            files = ids
        else:
            files = self.ids()
        for id in tqdm(files):
            self.data[id] = self.__get(id)
            
    def __get(self, id):
        data = pq.read_table(f'{self.DATA_PATH}/{id}.parquet')
        if self.formators is not None:
            for formator in self.formators:
                data = formator(data)
        return data


class PhotoReader(Reader):
    # Load data directly from local jpg files
    def __init__(self, 
                 DATA_PATH: str,
                 in_Memory: bool=False,
                 formators: list[Callable] = None,
                 img_format: str='jpg'):
        super().__init__(
            DATA_PATH = DATA_PATH,
            in_Memory = in_Memory
        )
        self.formators = formators
        self.img_format=img_format
        if in_Memory:
            self.load_data()
    
    def save_local(self, target_folder):
        for id in tqdm(self.ids()):
            data = self.get(id)
            file_path = f"{target_folder}/{id}.jpg"
            if os.path.exists(file_path):
                print(f"{id}.jpg file already exist!")
            else:
                data.save(file_path)
                
    def get(self, id):
        if self.in_Memory:
            return self.data[id]
        else:
            return self.__get(id)

    def load_data(self, ids = None):
        self.data = {}
        if ids is not None:
            files = ids
        else:
            files = self.ids()
        for id in tqdm(files):
            self.data[id] = self.__get(id)
            
    def __get(self, id):
        data = Image.open(f'{self.DATA_PATH}/{id}.{self.img_format}')
        data = np.array(data)
        if self.formators is not None:
            for formator in self.formators:
                data = formator(data)
        return data
    
    def ids(self):
        # This function should return the list of IDs based on the files in the DATA_PATH
        return [os.path.splitext(f)[0] for f in os.listdir(self.DATA_PATH) if f.endswith('.'+self.img_format)]