import torch

class MyPredictionClass:
    def __init__(self, cls: int, box: torch.Tensor, score: float, iou: torch.Tensor = None):
        self.cls = cls
        self.box = box
        self.score = score
        self.iou = iou

    def __repr__(self):
        return (f"YoloResultData(cls={self.cls}, box={self.box.tolist()}, "
                f"score={self.score:.4f}, iou={self.iou.tolist() if self.iou is not None else None})")
