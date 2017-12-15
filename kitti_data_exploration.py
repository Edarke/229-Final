import os.path
from glob import glob

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

image_paths = glob(os.path.join("data", "data_road", 'training', 'gt_image_2', '*.png'))
road = 0
total = 0.
iou = []
for image_file in image_paths:
    image = scipy.misc.imread(image_file)[:, :, 2]
    road = np.count_nonzero(image)
    total = image.shape[0] * image.shape[1]
    iou.append((total - road) / (2*total))
    print("% Road:", road/total, "% Not Road", 1 - road/total, 'total', total)
print('Done.')

print(np.average(iou))

road /= total
not_road = 1 - road

# % Road: 0.15315206373875018 % Not Road 0.8468479362612498 total 178168288.0


plt.bar(['Road', 'Not Road'], [road, not_road])
plt.title('Frequency of Kitti Classes')
plt.ylim((0, 1))
plt.show()


# Values from helper.calculate_class_frequency()
plt.bar(['void', 'flat', 'construction', 'object', 'nature', 'sky', 'human', 'vehicle'], [1.,  0.99193277,  0.99697479,  0.99327731,  0.97411765,  0.90285714,  0.84033613,  0.98689076])
plt.title('Frequency of Categories in Training Set')
plt.ylim((0, 1))
plt.show()