from typing import Dict, Literal, Tuple
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

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
    """ Metric for computing mean IoU and accuracy """

    def __init__(self, num_classes: int = 3, id=""):
        """ Builds and updates a confusion matrix.
            Args: num_classes: number of label classes
        """
        super().__init__(id)
        self.matrix = torch.zeros(num_classes, num_classes)
        self.class_range = torch.arange(num_classes)

    @torch.no_grad()
    def add(self, preds: torch.Tensor, labels: torch.Tensor):
        """ Updates using predictions and ground truth labels
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
        """ Resets the confusion matrix, should be called before each epoch """
        self.matrix.zero_()

    def compute_step(self) -> dict[str, float]:
        """ Computes the mean IoU and accuracy """
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
    
    def compute_rates(self) -> Dict[str, Tuple]:
        """ Returns: {
            "last": ndarray (n classes,)
            "running"       (epochs, n_classes)
        }
        Computes True-Positive, True-Negative, False-Positive, and False-Negative rates
        for the last epoch, and 
        """

        last = self.matrix.float().cpu().numpy()
        fTP = last.diagonal(axis1=1).astype(float)
        fFN = last.sum(axis=1) - fTP
        fFP = last.sum(axis=0) - fTP
        fTN = last.sum - (fTP + fFN + fFP)

        fTPR = fTP / (fTP + fFN + 1e-6)
        fTNR = fTN / (fTN + fFP + 1e-6)
        fFPR = fFP / (fFP + fTN + 1e-6)
        fFNR = fFN / (fTP + fFN + 1e-6)
        
        rec = torch.stack([ r["matrix"].float() for r in self.record ], dim=0).numpy()
        
        rTP = rec.diagonal(rec, axis1=1, axis2=2)
        rFN = rec.sum(axis=2) - rTP
        rFP = rec.sum(axis=1) - rTP
        rTN = rec.sum(axis=(1, 2)) - (rTP + rFN + fFP)

        rTPR = rTP / (rTP + rFN + 1e-6)
        rTNR = rTN / (rTN + rFP + 1e-6)
        rFPR = rFP / (rFP + rTN + 1e-6)
        rFNR = rFN / (rTP + rFN + 1e-6)

        self.last_cm = (fTPR, fTNR, fFPR, fFNR)
        self.runn_cm = (rTPR, rTNR, rFPR, rFNR)

        return {
            "last": self.last_cm,
            "running": self.runn_cm
        }
    
    def plot(self,
        type: Literal["last", "running", "f1", "recall", "precision"] = "last",
        title="",
        save_path = None
    ):
        (fTPR, fTNR, fFPR, fFNR) = self.last_cm


        # ---------- LAST CONFUSION MATRIX ----------
        last = self.matrix.float().cpu().numpy()  # shape (K, K)
        K = last.shape[0]
        classes = np.arange(K)

        # Helper: per-class metrics from a (K, K) confusion matrix
        def _per_class_metrics(cm: np.ndarray):
            tp = np.diag(cm).astype(float)
            fn = cm.sum(axis=1) - tp
            fp = cm.sum(axis=0) - tp

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            return precision, recall, f1

        precision_last, recall_last, f1_last = _per_class_metrics(last)

        if type in ["last", "precision", "recall", "f1"]:
            # 2D heatmap
            

            if type == "last":
                data = last
                x_label = "Predicted class"
                y_label = "True class"
                default_title = "Confusion matrix (last epoch)"
            else:
                if type == "precision":
                    data = precision_last[None, :]   # shape (1, K)
                    y_label = "Precision"
                    default_title = "Per-class precision (last epoch)"
                elif type == "recall":
                    data = recall_last[None, :]
                    y_label = "Recall"
                    default_title = "Per-class recall (last epoch)"
                else:  # "f1"
                    data = f1_last[None, :]
                    y_label = "F1 score"
                    default_title = "Per-class F1 (last epoch)"

                x_label = "Class index"

            plt.figure(figsize=(6, 5))

            im = plt.imshow(data, aspect="auto")
            plt.colorbar(im)

            plt.xticks(ticks=np.arange(K), labels=classes)
            if type == "last":
                plt.yticks(ticks=np.arange(K), labels=classes)
            else:
                plt.yticks(ticks=[0], labels=[y_label])

            plt.xlabel(x_label)
            plt.ylabel(y_label if type == "last" else "")
            plt.title(title or default_title)

        elif type == "running":
            # 3D surface of per-class F1 over epochs

            rec = torch.stack([r["matrix"].float() for r in self.record], dim=0).cpu().numpy()
            E = rec.shape[0]
            epochs = np.arange(E)

            # Compute per-class F1 for each epoch → (E, K)
            f1_epochs = np.zeros((E, K), dtype=float)
            for e in range(E):
                _, _, f1_e = _per_class_metrics(rec[e])
                f1_epochs[e] = f1_e

            # Meshgrid for surface
            X, Y = np.meshgrid(epochs, classes, indexing="ij")

            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(X, Y, f1_epochs, cmap="viridis")
            fig.colorbar(surf, shrink=0.5, aspect=10)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Class index")
            ax.set_zlabel("F1 score")
            ax.set_title(title or "Per-class F1 over epochs")

        else:
            raise ValueError(f"Unknown plot type: {type}")

        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()