from data.Dataloader import COCOHuamnDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

if __name__ == "__main__":
    dataDir = '/media/weiwang/Elements/coco'
    dataType = 'val2017'
    dataset = COCOHuamnDataset(dataDir, dataType)
    iter(DataLoader(dataset)).next()
