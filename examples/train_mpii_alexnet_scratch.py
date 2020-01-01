import sys
import shlex
import os.path
'''2019.12.31修改需要加上下列code才能够读取到scripts,否则就会报错的'''
import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
import scripts.train
import scripts.config

'''
#batch_size = 128 可能太大了,会导致GPU不够用,不妨改小一点
#本人GPU Geforce GTX 1660Ti(6G)不够128 batch_size
:'
max_iter:最大迭代次数,原code是1000000，我改的小一点
snapshot_step:快照步骤
batch_size:原来是128，内存不够，改为64
'''

argv = """
--max_iter 10000 \
--batch_size 64 \
--snapshot_step 5000 \
--test_step 250 \
--log_step 2 \
--dataset_name mpii
--train_csv_fn {0}/datasets/mpii/train_joints.csv \
--test_csv_fn {0}/datasets/mpii/test_joints.csv \
--val_csv_fn '' \
--img_path_prefix {0}/datasets/mpii/images \
--should_downscale_images \
--downscale_height 400 \
--n_joints 16 \
--seed 1701 \
--im_size 227 \
--min_dim 6 \
--shift 0.1 \
--bbox_extension_min 1.0 \
--bbox_extension_max 1.2 \
--coord_normalize \
--fname_index 0 \
--joint_index 1 \
--ignore_label -100500 \
--symmetric_joints "[[12, 13], [11, 14], [10, 15], [2, 3], [1, 4], [0, 5]]" \
--conv_lr 0.0005 \
--fc_lr 0.0005 \
--fix_conv_iter 0 \
--optimizer adagrad \
--o_dir {0}/out/mpii_alexnet_scratch \
--gcn \
--fliplr \
--workers 4 \
--net_type Alexnet \
--reset_iter_counter
""".format(scripts.config.ROOT_DIR)

argv = shlex.split(argv)
print(argv)
scripts.train.main(argv)
