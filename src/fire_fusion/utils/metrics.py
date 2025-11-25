from typing import Dict
import torch
import torch.nn as nn

class TemperatureScaler(nn.Module):
    """
    Temperature scaling: logit_scaled = logit / T
    T > 0; we optimize log_T for stability: T = exp(log_T).
    """
    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))  # log T, init T=1

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self.log_temperature)
        return logits / temperature


class Metric:
    def __init__(self, id = ""):
        self.id = id
        self.record = []

    def reset(self) -> None:
        raise NotImplementedError
    
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        raise NotImplementedError

    def compute_step(self) -> Dict:
        raise NotImplementedError


class Accuracy(Metric):
    def __init__(self, id = ""):
        super().__init__(id)
        self.correct_ign = 0
        self.correct_cause = 0

    def reset(self):
        """ should be called before each epoch """
        self.correct = 0
        self.correct = 0

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Updates accuracy and confusion matrix predictions and ground truth labels

        Args:
            preds (torch.LongTensor): (b,) or (b, h, w) tensor with class predictions
            labels (torch.LongTensor): (b,) or (b, h, w) tensor with ground truth class labels
        """
        # Accuracy
        self.correct += (preds.type_as(labels) == labels).sum().item()
        self.total += labels.numel()

    def compute_step(self) -> dict[str, float]:
        """ Return scores for the epoch, reset internal state, and update p/epoch record """
        acc = self.correct / (self.total + 1e-6)
        self.record.append(acc)

        scores = {
            f"accuracy{'_'+self.id if self.id else ''}": acc,
            f"n_samples{'_'+self.id if self.id else ''}": self.total,
        }
        self.reset()
        return scores


class ConfusionMatrix(Metric):
    """
    Metric for computing mean IoU and accuracy
    """

    def __init__(self, num_classes: int = 3, id=""):
        """
        Builds and updates a confusion matrix.

        Args:
            num_classes: number of label classes
        """
        super().__init__(id)
        self.matrix = torch.zeros(num_classes, num_classes)
        self.class_range = torch.arange(num_classes)

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Updates using predictions and ground truth labels

        Args:
            preds (torch.LongTensor): (b,) or (b, h, w) tensor with class predictions
            labels (torch.LongTensor): (b,) or (b, h, w) tensor with ground truth class labels
        """
        if preds.dim() > 1:
            preds = preds.view(-1)
            labels = labels.view(-1)

        preds_one_hot = (preds.type_as(labels).cpu()[:, None] == self.class_range[None]).int()
        labels_one_hot = (labels.cpu()[:, None] == self.class_range[None]).int()
        update = labels_one_hot.T @ preds_one_hot

        self.matrix += update

    def reset(self):
        """
        Resets the confusion matrix, should be called before each epoch
        """
        self.matrix.zero_()

    def compute_step(self) -> dict[str, float]:
        """
        Computes the mean IoU and accuracy
        """
        true_pos = self.matrix.diagonal()
        class_iou = true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)
        mean_iou = class_iou.mean().item()
        accuracy = (true_pos.sum() / (self.matrix.sum() + 1e-5)).item()

        self.record.append({
            "mean_iou": mean_iou,
            "accuracy": accuracy,
            "matrix": self.matrix,
        })

        return {
            "iou": mean_iou,
            "accuracy": accuracy,
        }
    
    def compute_rates(self):
        """
        Returns
        Computes True-Positive, True-Negative, False-Positive, and False-Negative rates
        for the last epoch, and 
        """

        last = self.matrix.float()
        
        fTP, fTN, fFP, fFN = last[1,1]+1e-6, last[0,0]+1e-6, last[0,1]+1e-6, last[1,0]+1e-6
        fTPR = fTP / (fTP + fFN)
        fTNR = fTN / (fTN + fFP)
        fFPR = fFP / (fFP + fTN)
        fFNR = fFN / (fTP + fFN)
        
        rec = torch.stack([r["matrix"].float() for r in self.record], dim=0)
        rTP, rTN, rFP, rFN = rec[:, 1, 1], rec[:, 0, 0], rec[:, 0, 1], rec[:, 1, 0]
        rTPR = rTP / (rTP + rFN)
        rTNR = rTN / (rTN + rFP)
        rFPR = rFP / (rFP + rTN)
        rFNR = rFN / (rTP + rFN)

        return {
            "last": (fTPR, fTNR, fFPR, fFNR),
            "running": (rTPR, rTNR, rFPR, rFNR)
        }