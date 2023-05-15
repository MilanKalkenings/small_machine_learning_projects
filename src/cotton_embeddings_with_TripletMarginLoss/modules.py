import torch.nn
from torchvision.models import resnet18, ResNet18_Weights
from milankalkenings.deep_learning import Module


class ResNet(Module):
    def __init__(self, alpha: float, n_classes: int = 6, triplet_loss_margin: float = 10, embedding_size: int = 2):
        """

        :param n_classes:
        :param float alpha: value in [0, 1], determines how much loss pays attention to classification
        """
        super(ResNet, self).__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        head_ins = resnet.fc.in_features

        # replace head
        del resnet.fc  # delete old head
        self.resnet = resnet

        # add embedding head
        self.emb_head = torch.nn.Linear(in_features=head_ins, out_features=embedding_size)
        self.loss_triplet = torch.nn.TripletMarginLoss(margin=triplet_loss_margin)

        # add cls head
        self.cls_head = torch.nn.Linear(in_features=embedding_size, out_features=n_classes)
        self.loss_cls = torch.nn.CrossEntropyLoss()

        self.alpha = alpha

    @staticmethod
    def find_triplets(embedding: torch.Tensor, y: torch.Tensor):
        dist_matrix = torch.cdist(embedding, embedding, p=2)
        n = embedding.shape[0]

        anchor_indices = torch.arange(n, dtype=torch.long)
        positive_indices = torch.zeros(n, dtype=torch.long)
        negative_indices = torch.zeros(n, dtype=torch.long)

        for i in range(n):
            same_class_indices = torch.nonzero(y == y[i]).squeeze()
            diff_class_indices = torch.nonzero(y != y[i]).squeeze()

            positive_checkup = dist_matrix[i].clone()
            positive_checkup[diff_class_indices] = 0  # 0 = min dist
            positive_indices[i] = torch.argmax(positive_checkup)

            negative_checkup = dist_matrix[i].clone()
            negative_checkup[same_class_indices] = torch.max(negative_checkup)
            negative_indices[i] = torch.argmin(negative_checkup)
        return anchor_indices, positive_indices, negative_indices

    def forward(self, x, y):
        embedding = self.emb_head(self.resnet(x))

        # triplet loss
        anchor_indices, positive_indices, negative_indices = self.find_triplets(embedding=embedding, y=y)
        anchors = embedding[anchor_indices]
        positives = embedding[positive_indices]
        negatives = embedding[negative_indices]
        loss_triplet = self.loss_triplet(anchors, positives,  negatives)

        # cls loss
        scores = self.cls_head(embedding)
        loss_cls = self.loss_cls(scores, y)

        loss = (self.alpha * loss_cls) + ((1 - self.alpha) * loss_triplet)
        return {"scores": scores, "loss": loss, "embedding": embedding, "loss_cls": loss_cls, "loss_triplet": loss_triplet}

    def freeze_pretrained(self):
        for param in self.resnet.parameters():
            param.requires_grad = False

    def unfreeze_pretrained(self):
        for param in self.resnet.parameters():
            param.requires_grad = True
