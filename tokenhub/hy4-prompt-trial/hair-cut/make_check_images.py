"""Make visual-check images for hy4: mask estimate, red overlay, and crops per mask region.
Usage: python make_check_images.py  (CPU only, light)"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

SRC = "sera-elvish-v5-mylora-b_00200_.png"
WRP = "sera-elvish-v5-mylora-b_00200-wrapped.png"
a = Image.open(SRC).convert("RGB"); b = Image.open(WRP).convert("RGB")
A = np.asarray(a).astype(float); B = np.asarray(b).astype(float)

m = ((B - A).mean(2) > 8)
mask = Image.fromarray((m * 255).astype("uint8")).filter(ImageFilter.MedianFilter(5))
mask.save("mask_estimated.png")
M = np.asarray(mask) > 127

# red overlay on original
ov = A.copy(); ov[M] = ov[M] * 0.5 + np.array([255, 0, 0]) * 0.5
Image.fromarray(ov.astype("uint8")).save("overlay_mask_on_original.png")

# per-region crops (padded) so hy4 can inspect edges at higher zoom
lab, n = ndi.label(ndi.binary_dilation(M, iterations=6))
for i, sl in enumerate(ndi.find_objects(lab), 1):
    y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
    p = 40; box = (max(x0-p,0), max(y0-p,0), min(x1+p,a.width), min(y1+p,a.height))
    w = Image.new("RGB", ((box[2]-box[0])*2, box[3]-box[1]))
    w.paste(a.crop(box), (0, 0))
    w.paste(Image.fromarray(ov.astype("uint8")).crop(box), (box[2]-box[0], 0))
    w.save(f"region{i}_orig_vs_overlay.png")
print("regions:", n, "mask ratio: %.2f%%" % (M.mean()*100))
