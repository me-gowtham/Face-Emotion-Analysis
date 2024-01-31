from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

device = torch.device('cpu')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

class Face_Emotion_CNN(nn.Module):
    def __init__(self):
        super(Face_Emotion_CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.cnn6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.cnn7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.cnn1_bn = nn.BatchNorm2d(8)
        self.cnn2_bn = nn.BatchNorm2d(16)
        self.cnn3_bn = nn.BatchNorm2d(32)
        self.cnn4_bn = nn.BatchNorm2d(64)
        self.cnn5_bn = nn.BatchNorm2d(128)
        self.cnn6_bn = nn.BatchNorm2d(256)
        self.cnn7_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)
        self.dropout = nn.Dropout(0.3)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.pool1(self.cnn1_bn(self.cnn1(x))))
        x = self.relu(self.pool1(self.cnn2_bn(self.dropout(self.cnn2(x)))))
        x = self.relu(self.pool1(self.cnn3_bn(self.cnn3(x))))
        x = self.relu(self.pool1(self.cnn4_bn(self.dropout(self.cnn4(x)))))
        x = self.relu(self.pool2(self.cnn5_bn(self.cnn5(x))))
        x = self.relu(self.pool2(self.cnn6_bn(self.dropout(self.cnn6(x)))))
        x = self.relu(self.pool2(self.cnn7_bn(self.dropout(self.cnn7(x)))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.dropout(self.fc1(x)))
        x = self.relu(self.dropout(self.fc2(x)))
        x = self.log_softmax(self.fc3(x))
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def load_trained_model(model_path):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    return model

@app.route('/', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image_file = request.files['image']

        img = Image.open(image_file).convert("RGB")
        gray = cv2.cvtColor(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier(r'C:\Users\lucig\Downloads\haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray)
        
        if len(faces) == 0:
            predicted_emotion = 'No Human Face Detected'
            return jsonify(predicted_emotion), 200

        pred = -1  

        for (x, y, w, h) in faces:
            resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
            X = resize_frame / 256
            X = Image.fromarray(resize_frame)
            X = transform(X).unsqueeze(0)

            model_path = r'C:\Users\lucig\Downloads\FER_trained_model.pt'
            model = load_trained_model(model_path)
            with torch.no_grad():
                model.eval()
                log_ps = model.cpu()(X)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                pred = int(top_class.numpy())

        emotion_dict = {0: 'Neutral', 1: 'Happy', 2: 'Surprise', 3: 'Sad',
                        4: 'Angry', 5: 'Disgust', 6: 'Fear'}
        predicted_emotion = emotion_dict[pred]

        return jsonify(predicted_emotion), 200
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
