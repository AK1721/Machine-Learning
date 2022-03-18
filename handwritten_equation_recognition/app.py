from flask import Flask, render_template, request
from DigitsDetection import ExtractObjects
from Model import model
from torch import load, unsqueeze, max
from torchvision import transforms

app = Flask(__name__, template_folder='view')


@app.route('/', methods=['GET'])
def home_page():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    image_file = request.files['imagefile']
    image_path = "./images/"+image_file.filename
    image_file.save(image_path)
    model.load_state_dict(load('model/model.pt'))
    extractor = ExtractObjects(image_path)

    digits = extractor.getDigits()
    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((45, 45)),
        transforms.ToTensor()
    ])
    classes = ['+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    equ = ""
    for digi in digits:
        data = unsqueeze(transformer(digi), 0)
        _, pred = max(model(data), 1)
        equ += classes[pred]

    result = eval(equ)
    return render_template('index.html', result= result, equ= equ)


if __name__ == '__main__':
    app.run(port=3000, debug=True)