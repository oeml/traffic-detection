from models import Darknet
import utils.datasets as datasets

import torch
import torch.utils.data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # load model
    model = Darknet('config/yolov3-tiny.cfg').to(device)
    model.load_state_dict(torch.load('weights/yolov3-tiny.pth'))
    model.eval()
    # load dataset
    dataloader = torch.utils.data.DataLoader(
        datasets.ImageFolder('data/custom/images'),
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    # write eval function (from dataset)

    # quantization