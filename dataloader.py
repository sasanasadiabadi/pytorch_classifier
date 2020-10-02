import numpy as np
import cv2
import os 

from torch.utils.data import Dataset, DataLoader

class XrayDataset(Dataset):
    def __init__(self, data_path, augment=False, is_val=False):
        np.random.seed(42) 
        self.data_path = data_path
        self.lat_images = sorted([f for f in os.listdir(self.data_path) if (f.endswith('.png') and "Lateral" in f)])
        self.ap_images = sorted([f for f in os.listdir(self.data_path) if (f.endswith('.png') and "AP" in f)])
        val_size = 5 if is_val else 0
        self.val_lat = np.random.choice(self.lat_images, val_size, replace=False)
        self.trn_lat = [f for f in self.lat_images if f not in self.val_lat]
        self.val_ap = np.random.choice(self.ap_images, val_size, replace=False)
        self.trn_ap = [f for f in self.ap_images if f not in self.val_ap]

        self.trn_images = np.concatenate((self.trn_lat, self.trn_ap))
        self.val_images = np.concatenate((self.val_lat, self.val_ap))

        self.is_val = is_val
        self.augment = False if self.is_val else augment
        self.images = self.val_images if self.is_val else self.trn_images

        if self.augment:
            # horizental flipping augmentation on the fly! 
            self.images = np.repeat(self.images, 2) # make a copy of each image for augmentation
            self.images = [v if i%2==0 else v[:-4] + '_aug.png' for i, v in enumerate(self.images)]
        #np.random.shuffle(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        if not self.augment:
            image = cv2.imread(os.path.join(self.data_path, self.images[i]))
        else:
            imgname = self.images[i]
            if 'aug' in imgname:
                imgname = imgname.replace('_aug', '')
                image = cv2.imread(os.path.join(self.data_path, imgname))
                image = cv2.flip(image, 1)
            else:
                image = cv2.imread(os.path.join(self.data_path, imgname))

        if self.augment and np.random.randint(4)==0: # randomly rotate images (25%)
            theta = np.random.randint(-10, 10)
            image = self._rotate(image, theta)

        image = image/255.0
        image = np.transpose(image, (2, 0, 1))

        if "Lateral" in self.images[i]:
            label = np.array([0])
        elif "AP" in self.images[i]:
            label = np.array([1])
        
        return image, label
    
    def _rotate(self, image, theta):
        h, w, _ = image.shape
        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)

        image = cv2.warpAffine(image, M, (w, h))
        image = cv2.resize(image, (w, h))

        return image


if __name__ == "__main__":
    dataset = XrayDataset(data_path="./Train", augment=True, is_val=False)
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for idx, sample in enumerate(dataloader):
        img, lbl = sample
        img = img.numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        cv2.imshow('img', img)
        cv2.waitKey(0)
