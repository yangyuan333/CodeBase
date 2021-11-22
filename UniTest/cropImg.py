import numpy as np
import cv2
import os

path = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\supmat_results_all'
renderPath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\supmat_render'
savePath = r'H:\YangYuan\Code\cpp_program\seuvcl-codebase-master2\data\graphics\physdata\motionData\rectImg'
box = [109,171,537,537]
imgPath = 'H36MS11Walking_Camera02_00044'

img = cv2.imread(os.path.join(path, imgPath+'.jpg'))
cropImg = img[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])]
cv2.imshow('1', cropImg)
cv2.waitKey(0)
cv2.imwrite(os.path.join(savePath, imgPath+'.jpg'), cropImg)

img = cv2.imread(os.path.join(renderPath, imgPath+'_render'+'.jpg'))
cropImg = img[box[1]:(box[1]+box[3]), box[0]:(box[0]+box[2])]
cv2.imshow('1', cropImg)
cv2.waitKey(0)
cv2.imwrite(os.path.join(savePath, imgPath+'_render'+'.jpg'), cropImg)