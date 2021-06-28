import sys
sys.path.insert(1, 'D:/Python/face.evoLVe.PyTorch/')

from backbone import model_irse as mi
from util import extract_feature_v1 as ef
from scipy import spatial

backbone = mi.Backbone([112, 112], 50, 'ir')


res = ef.extract_feature(
    data_root = "D:/Python/face.evoLVe.PyTorch/data/test_Aligned",
    backbone = backbone,
    model_root = "D:/Python/face.evoLVe.PyTorch/data/checkpoint/backbone_ir50_ms1m_epoch120.pth",
    input_size = [112, 112], 
    batch_size = 1,
    device = 'cpu', 
)

print(res[0], res[1])

print(1 - spatial.distance.cosine(res[0], res[1]))

