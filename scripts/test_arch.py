import os
import sys

# Add PlotNeuralNet to the system path
sys.path.append(os.path.abspath('scripts/PlotNeuralNet'))
from pycore.tikzeng import *

def to_Box(name, s_filer, n_filer, offset, to, height, depth, width, fill, caption=" "):
    return r"""
\pic[shift={""" + offset + """}] at """ + to + """ 
    {Box={
        name=""" + name + """,
        caption=""" + caption + r""",
        xlabel={{""" + str(n_filer) + """, }},
        zlabel=""" + str(s_filer) + """,
        fill=""" + fill + """,
        height=""" + str(height) + """,
        width=""" + str(width) + """,
        depth=""" + str(depth) + """
        }
    };
"""

print(to_Box("input_img", "224", "3", "(0,0,0)", "(0,0,0)", 32, 32, 1.5, "\\InputColor", "NAC Tile"))
