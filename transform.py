import cv2
import numpy as np
import yaml

frame = cv2.imread("frame_0.png", cv2.IMREAD_COLOR)
frame_size = (frame.shape[1], frame.shape[0])

#Coordinates of quadrangle vertices in the source image.
src = np.float32([
                [450,230],
                [600,312],
                [80,312],
                [281,230]])

#Coordinates of the corresponding quadrangle vertices in the destination image.
dst = np.float32([
        [450,230],
        [450,312],
        [281,312],
        [281,230]])

#Calculates a perspective transform matrix from four pairs of the corresponding points.
H = cv2.getPerspectiveTransform(src, dst)
#Applies a perspective transformation to an image.
warped = cv2.warpPerspective(frame, H, frame_size, flags=cv2.INTER_LINEAR)

# create indices of the destination image and linearize them
h, w = warped.shape[:2]
indy, indx = np.indices((h, w), dtype=np.float32)
lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])

# warp the coordinates of warped to those of frame
map_ind = H.dot(lin_homg_ind)
map_x, map_y = map_ind[:-1]/map_ind[-1]
map_x = map_x.reshape(h, w).astype(np.float32)
map_y = map_y.reshape(h, w).astype(np.float32)

# # remap!
# dst = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
# blended = cv2.addWeighted(warped, 0.5, dst, 0.5, 0)
# cv2.imshow('blended.png', blended)
# cv2.waitKey()
# cv2.destroyAllWindows()

#Dump map_x and map_y in yaml file (remap.yaml)
fname = "remap.yaml"
data = {"remap_ipm_x":{"rows":frame.shape[1],"cols":frame.shape[0],"data":map_x.tolist()},"remap_ipm_y":{"rows":frame.shape[1],"cols":frame.shape[0],"data":map_y.tolist()}}
with open(fname, "w") as f:
    yaml.dump(data, f)

#Examining the remap file generated with the yaml file provided by the tuSimple dataset
ipm_remap_file_path ='./tusimple_ipm_remap.yml' 
#ipm_remap_file_path ='./remap.yaml'

fs = cv2.FileStorage(ipm_remap_file_path, cv2.FILE_STORAGE_READ)

remap_to_ipm_x = fs.getNode('remap_ipm_x').mat()
remap_to_ipm_y = fs.getNode('remap_ipm_y').mat()

print (remap_to_ipm_x)


