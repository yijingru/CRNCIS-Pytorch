
batch_size = 36
init_lr = 0.001
num_epochs = 500
milestones = [400, 450]

num_classes = 2
root = 'root'
img_sizes = (640, 512)   # (width, height)
output_dir = 'output'

variances = [0.1, 0.2]


top_k = 200
conf_thresh = 0.3
nms_thresh = 0.3


labelmap = ['cell']