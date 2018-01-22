import argparse
from utils import *
import pickle
from docopt import docopt
import time

docstr = """Find semantic segmentation metrics for given predictions and ground truth images(For PASCAL VOC 2012).

Usage:
  demo.py convert_prediction <predict_path> <id_file>
  demo.py convert_gt <gt_path> <id_file>
  demo.py find_metrics <predict_path> <gt_path> <id_file> [options]
  demo.py (-h | --help)
  demo.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --batch_size=<int>          batch_size for processing in gpu[default: 20]
  --gpu=<int>                 if GPU is to be used, and which GPU to be used. Set blank for no GPU.[default: 0]
  --classes=<int>             number of classes[default: 21]
  --ignore_label=<int>        label to be ignored[default: 255]

"""
if __name__ == '__main__':
    args = docopt(docstr, version='v0.1')
    if(args['--gpu']):
        torch.cuda.set_device(int(args['--gpu']))
    start_time = time.time()
    if(args['find_metrics']):
        ###########
        pascal_voc_util =  pickle.load( open( "pascal_voc_util.pkl", "rb" ) )
        class_dict = pascal_voc_util[1]
        class_names = [class_dict[i] for i in class_dict.keys()][:-1]
        ###########
        file_ids = get_file_ids(args['<id_file>'])
        st = time.time()
        print('Making Histogram')
        hist = hist_maker(args['<predict_path>'],args['<gt_path>'],file_ids,
                          int(args['--batch_size']),int(args['--ignore_label']),int(args['--classes']),bool(args['--gpu']))
        print('Histogram Made!')
        print(time.time()-st)
        mean_acc = mean_pixel_accuracy(hist.numpy(),class_names)
        overall_acc = pixel_accuracy(hist.numpy())
        mean_ious = mean_iou(hist.numpy(),class_names)
        fmiou = freq_weighted_miou(hist.numpy(), class_names)
        print('Total time taken: ',time.time()-start_time)
        
    if(args['convert_prediction']):
        ########
        pascal_voc_util =  pickle.load( open( "pascal_voc_util.pkl", "rb" ) )
        color_map = pascal_voc_util[0]
        ########
        file_ids = get_file_ids(args['<id_file>'])
        mat_to_png(args['<predict_path>'],file_ids,color_map)
        print('Total time taken: ',time.time()-start_time)
        
    if(args['convert_gt']):
        ########
        pascal_voc_util =  pickle.load( open( "pascal_voc_util.pkl", "rb" ) )
        color_map = pascal_voc_util[0]
        ########
        file_ids = get_file_ids(args['<id_file>'])
        img_dim_reductor(args['<gt_path>'],file_ids,color_map)
        print('Total time taken: ',time.time()-start_time)