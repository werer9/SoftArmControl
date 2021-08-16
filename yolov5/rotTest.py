import torch
import cv2
import numpy as np

from PIL import Image
import torchvision.transforms.functional as TF


class rotCNN(object):
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        PATH = "./models/rotation"
        self.model = torch.load(PATH).to(device)
        self.model.eval()
        print("LOADED rotationCNN")
    
    def classify(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((224,224))

        image = np.array(image)
        x = TF.to_tensor(image).cuda()
        x.unsqueeze_(0)
            
        output = self.model(x)
        output = output[0]
        pred = list(output.cpu().detach())

        if pred[0] < 0.143:
            angle = -75 + (20 * (pred[0]/0.143))
        elif pred[0] < 0.286:
            angle = -55 + (10 * (pred[0]/0.286))
        elif pred[0] < 0.429:
            angle = -45 + (10 * (pred[0]/0.429))
        elif pred[0] < 0.572:
            angle = -35 + (10 * (pred[0]/0.572))
        elif pred[0] < 0.715:
            angle = -25 + (10 * (pred[0]/0.715))
        elif pred[0] < 0.858:
            angle = -15 + (7.5 * (pred[0]/0.858))
        elif pred[0] < 1:
            angle = -7.5 + (7.5 * (pred[0]/1))
        elif pred[0] < 1.143 :
            angle = 0 + (7.5 * (pred[0]/1.143))
        elif pred[0] < 1.286:
            angle = 7.5 + (7.5 * (pred[0]/1.286))
        elif pred[0] < 1.429:
            angle = 15 + (10 * (pred[0]/1.429))
        elif pred[0] < 1.572:
            angle = 25 + (10 * (pred[0]/1.572))
        elif pred[0] < 1.715:
            angle = 35 + (10 * (pred[0]/1.715))
        elif pred[0] < 1.858:
            angle = 45 + (10 * (pred[0]/1.858))
        else:
            angle = 55 + (20 * (pred[0]/2))
        return float(angle)
