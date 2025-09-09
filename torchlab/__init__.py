import os
import torch
import glob

candidates = glob.glob(os.path.join(os.path.dirname(__file__), "torchlab_C*.so"))

if candidates:
    torch.ops.load_library(candidates[0])
    customsigmoid = torch.ops.torchlab.customsigmoid
else:
    customsigmoid = None
