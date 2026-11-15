"""Build inpaint masks (hy4 region numbering: left->right by center).
mode=delete : replace everything under the mask (grow/feather from hy4 §7 'intentional deletion')
mode=bgonly : hy4 per-region shrink -> grow -> feather (assumes mask touches character)
Outputs masks/region{1..4}.png and masks/all.png (white = repaint, soft edges)."""
import sys, os
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

MODE = sys.argv[1] if len(sys.argv) > 1 else "delete"
SRC = "sera-elvish-v5-mylora-b_00200_.png"
WRP = "sera-elvish-v5-mylora-b_00200-wrapped.png"
a = np.asarray(Image.open(SRC).convert("RGB")).astype(float)
b = np.asarray(Image.open(WRP).convert("RGB")).astype(float)
raw = Image.fromarray((((b - a).mean(2) > 8) * 255).astype("uint8")).filter(ImageFilter.MedianFilter(5))
M = np.asarray(raw) > 127

lab, n = ndi.label(ndi.binary_dilation(M, iterations=6))
objs = ndi.find_objects(lab)
order = sorted(range(1, n + 1), key=lambda i: (objs[i-1][1].start + objs[i-1][1].stop) / 2)  # hy4 numbering

# (shrink, grow, feather) px at 832x1216, from hy4_eval §7
BG = {1: (10, 3, 4), 2: (12, 1, 3), 3: (8, 2, 4), 4: (10, 4, 7)}
DEL = (0, 6, 8)
PAD = 12

def edge_dilate(m, r):
    # treat canvas edge as continuing so clipped masks do not leave a seam
    p = np.pad(m, PAD, mode="edge")
    p = ndi.binary_dilation(p, iterations=r) if r > 0 else p
    return p[PAD:-PAD, PAD:-PAD]

os.makedirs("masks", exist_ok=True)
total = np.zeros(M.shape, float)
for k, i in enumerate(order, 1):
    r = (lab == i) & M
    shrink, grow, feather = (DEL if MODE == "delete" else BG[k])
    if shrink: r = ndi.binary_erosion(r, iterations=shrink)
    r = edge_dilate(r, grow)
    im = Image.fromarray((r * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(feather / 2))
    im.save(f"masks/region{k}.png")
    total = np.maximum(total, np.asarray(im, float))
    print(f"region{k}: shrink={shrink} grow={grow} feather={feather} area={int(r.sum())}px")
Image.fromarray(total.astype("uint8")).save("masks/all.png")
