import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights as weights


def init_model():
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights.COCO_V1)


def predict_boxes(input_tensor: torch.Tensor, model: torch.nn.Module, thresh: float):
    model.eval()
    output = model(input_tensor)[0]
    print(output["scores"])

    boxes = output["boxes"]
    scores = output["scores"]
    return boxes[:(scores > thresh).sum().item()]
    
    
def epoch_loss(imgs: list, boxes_gt: list, model, objectness_weight: float, rpn_weight: float, device: str):
    losses = []
    with torch.no_grad():
        for i in range(len(imgs)):
            img = imgs[i].to(device)
            labels = torch.zeros(size=[len(boxes_gt[i])], dtype=torch.int64)  
            target = [{"boxes": torch.tensor(boxes_gt[i]).to(device), 
                    "labels": labels.to(device)}]
            out = model(img, target)
            loss = out["loss_box_reg"] + objectness_weight * out["loss_objectness"] + rpn_weight * out["loss_rpn_box_reg"]
            losses.append(loss.item())
    return losses






