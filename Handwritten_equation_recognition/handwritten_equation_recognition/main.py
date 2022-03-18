from DigitsDetection import ExtractObjects
from Model import model
import torch
from torchvision import transforms

if __name__ == '__main__':
    device = torch.device("cuda:0")
    model.to(device)
    model.load_state_dict(torch.load('model/model.pt'))
    extractor = ExtractObjects("equ.jpg")
    digits = extractor.getDigits()
    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((45, 45)),
        transforms.ToTensor()
    ])
    classes = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    equ = ""
    for digi in digits:
        data = torch.unsqueeze(transforms(digi), 0)
        data = data.to(device)
        _, pred = torch.max(model(data), 1)
        equ += classes[pred]

    print(equ)




