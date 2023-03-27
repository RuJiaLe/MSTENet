import argparse

parser = argparse.ArgumentParser()

# parameters
parser.add_argument('--total_epoch', type=int, default=20, help='epoch number')
parser.add_argument('--train_type', type=str, default='video', help='train_type')

# lr_decay
parser.add_argument('--base_lr', type=float, default=1e-5, help='base learning rate')
parser.add_argument('--final_lr', type=float, default=1e-6, help='final learning rate')

# step
parser.add_argument('--weight_decay', type=float, default=5e-5, help='base_decay rate of weight')
parser.add_argument('--size', type=int, default=224, help='training dataset size')
parser.add_argument('--num_workers', type=int, default=12, help='number of data loading workers.')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument("--clip_len", type=int, default=4, help="the number of frames in a video clip.")
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--log_dir', type=str, default="./Log", help="log_dir file")

# data_path
parser.add_argument('--train_path', type=str, default="./data/train_data", help='train_path')
parser.add_argument('--test_path', type=str, default="./data/test_data", help='test_path')
parser.add_argument('--result_path', type=str, default="./result", help='result_path')

# dataset
parser.add_argument('--train_dataset', type=list, default=["DAVIS_30", "DAVSOD_61"],
                    choices=["DUTS_TR", "HKU_TR", "DAVIS_30", "DAVSOD_61", "UVSD_12"],
                    help='train_dataset')

parser.add_argument('--test_dataset', type=list, default=["DAVIS_20"],
                    choices=["DUTS_TE", "HKU_TE", "DAVIS_20", "Validation_Set_46", "UVSD_6",
                             "Difficult-20", "Easy-35", "Normal-25", "SegV2_13", "ViSal_17", "VOS_40"],
                    help='predict_dataset')

# model_path·
parser.add_argument('--model_path', type=str, default="./save_model", help='save_model_path')

args = parser.parse_args()
