import torch

class MyPredictionClass:
    def __init__(self, cls: int, box: torch.Tensor, score: float, iou: float = 0):
        self.cls = cls
        self.box = box
        self.score = score
        self.iou = iou

    def __repr__(self):
        return (f"(detected object ("
                # f"cls={self.cls}, "
                f"box={self.box.tolist()}, "
                f"score={self.score:.4f}, "
                f"iou={self.iou:.4f})"
                )
