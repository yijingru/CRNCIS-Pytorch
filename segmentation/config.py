
batch_size = 10
init_lr = 0.001
num_epochs = 140
num_classes = 2
root = 'root'
img_sizes = (640, 512)   # (w, h)
output_dir = 'output'

variances = [0.1, 0.2]


top_k = 200
conf_thresh = 0.3
nms_thresh = 0.3
seg_thresh = 0.4

labelmap = ['cell']
