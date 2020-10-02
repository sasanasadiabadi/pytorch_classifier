import numpy as np
import cv2
import os 
import argparse
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from model import Classifier
from mobilenetV1 import MobilenetV1
from dataloader import XrayDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class ImgClassification(object):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epochs, batchsize, output_model):
        # setup model
        torch.manual_seed(42) # for reproducibility! 
        model = Classifier()
        model = model.to(device=self.device)   

        # setup data loader 
        trn_dataset =  XrayDataset(data_path="./Train", augment=True)  
        val_dataset = XrayDataset(data_path='./Test', augment=False)
        trn_loader = DataLoader(trn_dataset, batch_size=batchsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)

        # setup loss
        criterion = nn.BCELoss()

        # setup optimizer 
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8)

        # setup trainer 
        best_loss = np.inf 
        trn_loss = []
        val_loss = []
        model.train()
        for epoch in range(epochs):
            pbar = tqdm(desc='Epoch {:02d}/{}'.format(epoch+1, epochs), total=len(trn_loader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

            losses = []
            for idx, sample in enumerate(trn_loader):
                image, label = sample
                image, label = image.to(self.device).float(), label.to(self.device).float()

                optimizer.zero_grad()
                pred_label = model(image)
                loss = criterion(pred_label, label)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                pbar.update(1)

            trn_loss.append(np.mean(losses))

            # evaluate model on val set 
            model.eval()
            with torch.no_grad():
                val_losses = []
                for idx, sample in enumerate(val_loader):
                    image, label = sample
                    image, label = image.to(self.device).float(), label.to(self.device).float()

                    pred_label = model(image)
                    loss = criterion(pred_label, label)
                    val_losses.append(loss.item())
                val_loss.append(np.mean(val_losses))
            
            pbar.set_postfix(trn_loss=np.mean(losses), val_loss=np.mean(val_losses))
            pbar.close()

            # save model
            torch.save(model.state_dict(), output_model)

            model.train()
            scheduler.step()

        # plot the training curves 
        plt.plot(np.arange(epochs), trn_loss, 'b-', label='train')
        plt.plot(np.arange(epochs), val_loss, 'r-', label='validation')
        plt.legend(loc=1)
        plt.show()

    def predict(self, img_path, model_path, visualize=False):
        model = Classifier()
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt)
        model = model.to(self.device)
        model.eval()

        images = [f for f in os.listdir(img_path) if f.endswith('.png')]
        success = 0 
        y_true, y_pred = [], []
        for imgname in images:
            image = cv2.imread(os.path.join(img_path, imgname))
            img = image.copy()/255.0 
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            img = torch.from_numpy(img).to(self.device).float()

            pred_label = model(img).cpu().detach().numpy()
            
            if ("Lateral" in imgname) and (pred_label < 0.5):
                success += 1
            elif ("AP" in imgname) and (pred_label > 0.5):
                success += 1

            true_label = "Lateral" if "Lateral" in imgname else "AP"
            pred_label = "Lateral" if pred_label < 0.5 else "AP"
            y_true.append(true_label)
            y_pred.append(pred_label)
            if visualize:
                cv2.putText(image, "pred: {}".format(pred_label), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                cv2.putText(image, "true: {}".format(true_label), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                cv2.imshow('predicted', image)
                cv2.waitKey(0)
        
        print("[INFO] accuracy is : {:.2f} %".format(100*success/len(images)))
        cm = confusion_matrix(y_true, y_pred)
        print(classification_report(y_true, y_pred, target_names=["AP", "Lateral"]))
        #print("[INFO] confusion matrix:\n {}".format(cm))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=("AP", "Lateral")).plot()
        plt.show()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="SnkOs challenge parser!")
    subparser = parser.add_subparsers(dest="command", help='train/predict subparser')

    trn_parser = subparser.add_parser("train")
    trn_parser.add_argument("--epochs", help="number of epochs", type=int, default=20)
    trn_parser.add_argument("--batchsize", help="number of batches", type=int ,default=4)
    trn_parser.add_argument("--output_model", help="path to save model directory", type=str, default="models/model.pth")

    pred_parser = subparser.add_parser("predict")
    pred_parser.add_argument("--img_path", help="path to images directory", type=str, default="./Test")
    pred_parser.add_argument("--model_path", help="path to trained model", type=str, default="models/model.pth")
    pred_parser.add_argument("--vis", help="set 1 to visualize predictions", action="store_true")
    # parse arguments 
    args = parser.parse_args()
    
    # instantiate classifier 
    clf = ImgClassification()
    
    if args.command == "train":
        clf.train(args.epochs, args.batchsize, args.output_model)
    elif args.command == "predict":
        clf.predict(img_path=args.img_path, model_path=args.model_path, visualize=args.vis)
