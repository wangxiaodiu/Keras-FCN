import os
import numpy as np

if __name__ == '__main__':
    print("Turn all annotations to png. Python 3.6 required.")
    path_prefix = '/home/niu/Liang_Niu3/IIT_Affordances_2017'
    src_path = os.path.join(path_prefix, 'affordances_labels')
    dest_path = os.path.join(path_prefix, 'affordances_labels_png')
    print(f"Src:{src_path}, Dest:{dest_path}")
    with open(os.path.join(path_prefix, 'labels.txt')) as f:
        all_labels = f.readlines()
    for idx,f in enumerate(all_labels):
        f = os.path.join(path_prefix, f.strip('\n'))
        ann = np.loadtxt(f)
        print(idx, f, ", shape:", ann.shape)
        # TODO: trans ann to png
