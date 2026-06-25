"""
generate_hole_arch.py
======================
Generates a LaTeX TikZ file of the HOLE architecture using PlotNeuralNet,
compiles it using pdflatex, and converts the resulting PDF to a high-resolution PNG
using pdftoppm.
"""

import os
import sys
import subprocess
import shutil

# Add PlotNeuralNet to the system path
sys.path.append(os.path.abspath('scripts/PlotNeuralNet'))
from pycore.tikzeng import *

def get_custom_cor():
    return r"""
\def\InputColor{rgb:white,9;black,1.5}              % Input Image (Off-white)
\def\ConvColor{rgb:blue,2;green,5;white,10}         % DINOv3 Backbone (Light Blue)
\def\ConvReluColor{rgb:red,1;green,6;blue,6}        % LoRA Adapter (Teal/Cyan)
\def\PoolColor{rgb:orange,8;white,2}                % MLP Layer 1 (Orange)
\def\UnpoolColor{rgb:orange,5;white,5}              % MLP Layer 2 (Light Orange)
\def\FcColor{rgb:magenta,8;white,2}                 % Output Head (Magenta)
\def\SoftmaxColor{rgb:red,8;white,2}                % Loss Function (Red)
\def\SumColor{rgb:blue,5;green,15}
"""

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

def to_RightBandedBox(name, s_filer, n_filer, offset, to, height, depth, width, fill, bandfill, caption=" "):
    return r"""
\pic[shift={ """ + offset + """ }] at """ + to + """ 
    {RightBandedBox={
        name=""" + name + """,
        caption=""" + caption + """,
        xlabel={{ """ + str(n_filer[0]) + """, """ + str(n_filer[1]) + """ }},
        zlabel=""" + str(s_filer) + """,
        fill=""" + fill + """,
        bandfill=""" + bandfill + """,
        height=""" + str(height) + """,
        width={ """ + str(width[0]) + """ , """ + str(width[1]) + """ },
        depth=""" + str(depth) + """
        }
    };
"""

def main():
    # Define the HOLE model architecture diagram components
    # Spacing increased significantly to prevent horizontal overlap
    arch = [
        to_head('scripts/PlotNeuralNet'),
        get_custom_cor(),
        to_begin(),
        
        # 1. Input image (224x224x3)
        to_Box("input_img", " ", 3, offset="(0,0,0)", to="(0,0,0)", height=32, depth=32, width=1.5, fill="\\InputColor", caption=" "),
        
        # 2. Patch Projection (14x14x384)
        to_Box("patch_proj", " ", 384, offset="(4.0,0,0)", to="(input_img-east)", height=16, depth=16, width=2.0, fill="\\ConvColor", caption=" "),
        to_connection("input_img", "patch_proj"),
        
        # 3. DINOv3 Backbone + LoRA (12 blocks, 384 dim, rank 32 LoRA)
        to_RightBandedBox("vit_blocks", " ", ("384", "32"), offset="(4.0,0,0)", to="(patch_proj-east)", height=16, depth=16, width=(4.0, 1.5), fill="\\ConvColor", bandfill="\\ConvReluColor", caption=" "),
        to_connection("patch_proj", "vit_blocks"),
        
        # 4. [CLS] Token (1x1x384) - Drawn as a thin vector representation
        to_Box("cls_token", 1, 384, offset="(4.0,0,0)", to="(vit_blocks-east)", height=1.5, depth=1.5, width=4.0, fill="\\ConvColor", caption=" "),
        to_connection("vit_blocks", "cls_token"),
        
        # ── HEAD 1 (64d MLP) ──
        to_Box("h1_mlp1", 1, 192, offset="(4.5, 6.0, 0)", to="(cls_token-east)", height=1.5, depth=1.5, width=1.5, fill="\\PoolColor", caption=" "),
        to_connection("cls_token", "h1_mlp1"),
        to_Box("h1_mlp2", 1, 96, offset="(2.5, 0, 0)", to="(h1_mlp1-east)", height=1.5, depth=1.5, width=1.0, fill="\\UnpoolColor", caption=" "),
        to_connection("h1_mlp1", "h1_mlp2"),
        to_Box("h1_head", 1, 64, offset="(2.5, 0, 0)", to="(h1_mlp2-east)", height=1.5, depth=1.5, width=0.8, fill="\\FcColor", caption=" "),
        to_connection("h1_mlp2", "h1_head"),
        
        # ── HEAD 2 (128d MLP) ──
        to_Box("h2_mlp1", 1, 256, offset="(4.5, 2.0, 0)", to="(cls_token-east)", height=1.5, depth=1.5, width=2.0, fill="\\PoolColor", caption=" "),
        to_connection("cls_token", "h2_mlp1"),
        to_Box("h2_mlp2", 1, 128, offset="(2.5, 0, 0)", to="(h2_mlp1-east)", height=1.5, depth=1.5, width=1.5, fill="\\UnpoolColor", caption=" "),
        to_connection("h2_mlp1", "h2_mlp2"),
        to_Box("h2_head", 1, 128, offset="(2.5, 0, 0)", to="(h2_mlp2-east)", height=1.5, depth=1.5, width=1.5, fill="\\FcColor", caption=" "),
        to_connection("h2_mlp2", "h2_head"),
        
        # ── HEAD 3 (256d MLP) ──
        to_Box("h3_mlp1", 1, 512, offset="(4.5, -2.0, 0)", to="(cls_token-east)", height=1.5, depth=1.5, width=2.5, fill="\\PoolColor", caption=" "),
        to_connection("cls_token", "h3_mlp1"),
        to_Box("h3_mlp2", 1, 256, offset="(2.5, 0, 0)", to="(h3_mlp1-east)", height=1.5, depth=1.5, width=2.0, fill="\\UnpoolColor", caption=" "),
        to_connection("h3_mlp1", "h3_mlp2"),
        to_Box("h3_head", 1, 256, offset="(2.5, 0, 0)", to="(h3_mlp2-east)", height=1.5, depth=1.5, width=2.0, fill="\\FcColor", caption=" "),
        to_connection("h3_mlp2", "h3_head"),
        
        # ── HEAD 4 (384d MLP - Primary) ──
        to_Box("h4_mlp1", 1, 768, offset="(4.5, -6.0, 0)", to="(cls_token-east)", height=1.5, depth=1.5, width=3.0, fill="\\PoolColor", caption=" "),
        to_connection("cls_token", "h4_mlp1"),
        to_Box("h4_mlp2", 1, 384, offset="(2.5, 0, 0)", to="(h4_mlp1-east)", height=1.5, depth=1.5, width=2.5, fill="\\UnpoolColor", caption=" "),
        to_connection("h4_mlp1", "h4_mlp2"),
        to_Box("h4_head", 1, 384, offset="(2.5, 0, 0)", to="(h4_mlp2-east)", height=1.5, depth=1.5, width=2.5, fill="\\FcColor", caption=" "),
        to_connection("h4_mlp2", "h4_head"),
        
        # ── Loss Layers ──
        to_SoftMax("ntxent_loss", s_filer=" ", offset="(4.0, 0, 0)", to="(h2_head-east)", height=8, depth=8, width=2.0, caption=" "),
        to_connection("h1_head", "ntxent_loss"),
        to_connection("h2_head", "ntxent_loss"),
        to_connection("h3_head", "ntxent_loss"),
        to_connection("h4_head", "ntxent_loss"),
        
        to_SoftMax("hinge_loss", s_filer=" ", offset="(4.0, 0, 0)", to="(h4_head-east)", height=6, depth=6, width=2.0, caption=" "),
        to_connection("h4_head", "hinge_loss"),

        # Custom Z-Labels placed on the RIGHT bottom edge (Z-axis) at pos=0.5 (middle)
        r"\path (input_img-nearsoutheast) -- (input_img-farsoutheast) coordinate[pos=0.5] (input_img-zmid);",
        r"\node [anchor=north, sloped, xshift=3mm] at (input_img-zmid) {224};",
        
        r"\path (patch_proj-nearsoutheast) -- (patch_proj-farsoutheast) coordinate[pos=0.5] (patch_proj-zmid);",
        r"\node [anchor=north, sloped, xshift=3mm] at (patch_proj-zmid) {14};",
        
        r"\path (vit_blocks-nearsoutheast) -- (vit_blocks-farsoutheast) coordinate[pos=0.5] (vit_blocks-zmid);",
        r"\node [anchor=north, sloped, xshift=3mm] at (vit_blocks-zmid) {14};",

        # Custom captions securely placed 1.5cm below each box's southeast anchor to never overlap with dimensions
        r"\node [below=1.5cm of input_img-southeast, anchor=north, align=center] {\textbf{NAC Tile}};",
        r"\node [below=1.5cm of patch_proj-southeast, anchor=north, align=center] {\textbf{Patch Proj.}};",
        r"\node [below=1.5cm of vit_blocks-southeast, anchor=north, align=center] {\textbf{DINOv3 ViT-S/16 Backbone}\\\textbf{+ LoRA Adapter}};",
        r"\node [below=1.5cm of cls_token-southeast, anchor=north, align=center] {\textbf{[CLS] Token}};",
        
        # Loss Layers text (placed above and below)
        r"\node [above=0.8cm of ntxent_loss-north, align=center] {\textbf{NT-Xent Loss}};",
        r"\node [below=0.8cm of hinge_loss-south, align=center] {\textbf{Hinge Triplet Loss}};",
        
        # MLPs text
        r"\node [above=0.8cm of h1_mlp1-north, align=center] {\small MLP1};",
        r"\node [above=0.8cm of h1_mlp2-north, align=center] {\small MLP2};",
        r"\node [above=0.8cm of h1_head-north, align=center] {\small Head};",
        
        r"\node [above=0.8cm of h2_mlp1-north, align=center] {\small MLP1};",
        r"\node [above=0.8cm of h2_mlp2-north, align=center] {\small MLP2};",
        r"\node [above=0.8cm of h2_head-north, align=center] {\small Head};",
        
        r"\node [below=1.0cm of h3_mlp1-southeast, align=center] {\small MLP1};",
        r"\node [below=1.0cm of h3_mlp2-southeast, align=center] {\small MLP2};",
        r"\node [below=1.0cm of h3_head-southeast, align=center] {\small Head};",
        
        r"\node [below=1.0cm of h4_mlp1-southeast, align=center] {\small MLP1};",
        r"\node [below=1.0cm of h4_mlp2-southeast, align=center] {\small MLP2};",
        r"\node [below=1.0cm of h4_head-southeast, align=center] {\small Primary Head};",

        to_end()
    ]
    
    # Save LaTeX file
    tex_path = "scripts/generate_hole_arch.tex"
    print(f"Generating LaTeX architecture file: {tex_path}...")
    to_generate(arch, tex_path)
    
    # Compile LaTeX using pdflatex
    print("Compiling LaTeX file with pdflatex...")
    cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory=scripts", tex_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("LaTeX compilation failed!")
        print("Stdout:", res.stdout)
        print("Stderr:", res.stderr)
        sys.exit(1)
        
    pdf_path = "scripts/generate_hole_arch.pdf"
    if not os.path.exists(pdf_path):
        print(f"PDF was not generated at {pdf_path}!")
        sys.exit(1)
        
    # Convert PDF to high-resolution PNG using pdftoppm
    os.makedirs("figures", exist_ok=True)
    png_prefix = "figures/hole_architecture"
    print(f"Converting PDF to PNG with prefix: {png_prefix}...")
    cmd_ppm = ["pdftoppm", "-png", "-r", "150", pdf_path, png_prefix]
    res_ppm = subprocess.run(cmd_ppm, capture_output=True, text=True)
    if res_ppm.returncode != 0:
        print("pdftoppm conversion failed!")
        print("Stderr:", res_ppm.stderr)
        sys.exit(1)
        
    # Clean up LaTeX intermediate files
    print("Cleaning up intermediate compilation files...")
    for ext in [".aux", ".log", ".tex"]:
        fpath = f"scripts/generate_hole_arch{ext}"
        if os.path.exists(fpath):
            os.remove(fpath)
            
    print("Success! Generated figures/hole_architecture-1.png")

if __name__ == '__main__':
    main()
