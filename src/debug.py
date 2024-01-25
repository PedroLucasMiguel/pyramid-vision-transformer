import torch

from PIL import Image
from torchvision import transforms
from pvt_v2 import pvt_v2_b5

if __name__ == "__main__":

    IMG_NAME = "../samples/image.png"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    img = Image.open(IMG_NAME).convert("RGB")

    # Transformações da imagem de entrada
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = preprocess(img)
    input_batch = img.unsqueeze(0)
    input_batch = input_batch.to(DEVICE)

    model = pvt_v2_b5(2)
    model.to(DEVICE)

    print(model(input_batch).shape)