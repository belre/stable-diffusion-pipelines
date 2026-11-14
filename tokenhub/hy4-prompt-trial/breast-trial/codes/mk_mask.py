import cv2, numpy as np, sys
src = sys.argv[1] if len(sys.argv)>1 else "SAJ6O4sj.png"
im = cv2.imread(src); h,w = im.shape[:2]
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
m = cv2.inRange(hsv, (150,40,120), (179,255,255)) | cv2.inRange(hsv, (0,40,150), (5,255,255))
roi = np.zeros_like(m); roi[int(.27*h):int(.57*h), int(.30*w):int(.68*w)] = 255   # 顔・ベルト下を除外
m &= roi
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
cs,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(cs, key=cv2.contourArea); m[:] = 0; cv2.drawContours(m,[c],-1,255,-1)
m = cv2.erode(m, np.ones((5,5),np.uint8))          # 縁(輪郭線)を内側に外す
m = cv2.GaussianBlur(m,(15,15),0)
cv2.imwrite("mask.png", m)                          # グレースケール、白=編集範囲
ov = im.copy(); ov[m>128] = (0.5*ov[m>128] + 0.5*np.array([0,255,0])).astype(np.uint8)
cv2.imwrite("mask_overlay.png", ov[250:700,200:700])