import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path)
from spdnet.spd import SPDTangentSpace,SPDUnTangentSpace
from CRGNet import CRGNet,CRGNet_BCIC_2a,CRGNet_BCIC_2b,CRGNet_HGD,Simple_Net

