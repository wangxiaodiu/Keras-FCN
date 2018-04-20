import os
import sys
import numpy as np
from skimage.io import imsave
from scipy.ndimage.interpolation import zoom # as SCIPY_zoom

if __name__ == '__main__':
    print("Turn all annotations to png. Python >= 3.6 required.")
    assert sys.version_info[0] == 3 and sys.version_info[1] >= 6, "Python >= 3.6 required."
    path_prefix = '/home/niu/Liang_Niu3/IIT_Affordances_2017'
    src_path = os.path.join(path_prefix, 'affordances_labels')
    dest_path = os.path.join(path_prefix, 'affordances_labels_png')
    print(f"Src:{src_path}, Dest:{dest_path}")

    with open(os.path.join(path_prefix, 'labels.txt')) as f:
        all_labels = f.readlines()

    for idx,f in enumerate(all_labels):
        png_f = os.path.join(dest_path, os.path.splitext(os.path.basename(f))[0]+'.png')
        f = os.path.join(path_prefix, f.strip('\n'))
        ann = np.loadtxt(f,dtype=np.uint8)
        # print(ann.dtype)
        # break
        print(idx, '/', len(all_labels), f, ", shape:", ann.shape)
        w, h = ann.shape
        ann = zoom(ann, (512.0 / w, 512.0 / h), order=0)
        imsave(png_f, ann)
