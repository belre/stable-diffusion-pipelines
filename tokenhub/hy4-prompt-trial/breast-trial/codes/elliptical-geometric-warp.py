import cv2, numpy as np
TARGET="../../../../local/SAJ6O4sj.png"; OUT="out_warp.png"
CX,CY=0.50,0.41; RX,RY=0.18,0.16
SHRINK=0.18; DRAPE=0.06; FEATHER=31
im=cv2.imread(TARGET); h,w=im.shape[:2]
y,x=np.mgrid[0:h,0:w].astype(np.float32)
cx,cy=CX*w,CY*h; rx,ry=RX*w,RY*h
r2=((x-cx)/rx)**2+((y-cy)/ry)**2
wgt=np.exp(-2.5*r2).astype(np.float32)
wgt=cv2.GaussianBlur(wgt,(FEATHER|1,FEATHER|1),0)
map_x=x+SHRINK*wgt*(x-cx)                 # sample farther out => inward compression
map_y=y-DRAPE*wgt*np.maximum(y-cy,0)      # lower fabric drapes downward
warp=cv2.remap(im,map_x,map_y,cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
a=np.clip(wgt*1.6,0,1)[...,None]
out=(im.astype(np.float32)*(1-a)+warp.astype(np.float32)*a).clip(0,255).astype(np.uint8)
cv2.imwrite(OUT,out); print(OUT)