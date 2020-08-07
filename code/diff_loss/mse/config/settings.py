import os
import logging


#######################  网络参数设置　　############################################################
channel = 32
feature_map_num = 32
res_conv_num = 4   
unit_num = 32         
#scale_num = 4
num_scale_attention = 4
scale_attention = False
ssim_loss = True
########################################################################################
aug_data = False # Set as False for fair comparison

patch_size = 64
pic_is_pair = True  #input picture is pair or single

lr = 0.0005

data_dir = '/data1/datasets/haze_blur_rain/rain100H'
if pic_is_pair is False:
    data_dir = '/data1/wangcong/dataset/real-world-images'
log_dir = '../logdir'
show_dir = '../showdir'
model_dir = '../models'
show_dir_feature = '../showdir_feature'

log_level = 'info'
model_path = os.path.join(model_dir, 'latest_net')
save_steps = 400

num_workers = 8

num_GPU = 2

device_id = '1,2'

epoch = 1000
batch_size = 12

if pic_is_pair:
    root_dir = os.path.join(data_dir, 'train')
    mat_files = os.listdir(root_dir)
    num_datasets = len(mat_files)
    l1 = int(3/5 * epoch * num_datasets / batch_size)
    l2 = int(4/5 * epoch * num_datasets / batch_size)
    one_epoch = int(num_datasets/batch_size)
    total_step = int((epoch * num_datasets)/batch_size)

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


