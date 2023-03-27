import os
import torch
from model.model import Model
from dataload.dataload import VideoDataset
from torch.utils.data import DataLoader
from dataload.transforms import get_transforms
from Evaluation.Eval_utils import Eval_fmeasure, Eval_Smeasure, Eval_mae
from utils import Save_result
from tqdm import tqdm
import logging
from options import args

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_transforms = get_transforms(input_size=(args.size, args.size))
test_dataset = VideoDataset(root_dir=args.test_path, train_set_list=args.test_dataset, training=True,
                            transforms=test_transforms, clip_len=args.clip_len)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, drop_last=True, pin_memory=True)


def test(test_data, model, is_save_result=False):
    S_lambda = 0.0
    F_beta = 0.0
    Mae = 0.0

    model.eval()

    for packs in tqdm(test_data):

        with torch.no_grad():
            images, gts, paths = [], [], []

            for pack in packs:
                image, gt, path = pack['image'], pack['gt'], pack['path']
                images.append(image.to(device))
                gts.append(gt.to(device))
                paths.append(path)

            images = torch.cat(images, dim=0)
            gts = torch.cat(gts, dim=0)

            out = model(images)

            X, Y = [], []
            for i in range(args.clip_len):

                if is_save_result:
                    Save_result(out[i, :, :, :], paths[i][0])

                x = out[i, :, :, :].unsqueeze(0)
                y = gts[i, :, :, :].unsqueeze(0)

                X.append(x)
                Y.append(y)

            Mae += (Eval_mae(zip(X, Y))).data

            f, _, _ = Eval_fmeasure(zip(X, Y))
            F_beta += (f.max()).data

            S_lambda += (Eval_Smeasure(zip(X, Y))).data

    return S_lambda / len(test_data), F_beta / len(test_data), Mae / len(test_data)


if __name__ == '__main__':

    logging.basicConfig(filename=args.log_dir + '/test_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    model = Model()
    model.to(device)

    # 加载模型
    model_path = args.model_path + '/best_{}_model.pth'.format(args.train_type)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        print('*' * 20, 'Load best_model Done !!! ', '*' * 20)

    print('start test {} !!!'.format(args.test_dataset))

    S_lambda, F_beta, Mae = test(test_dataloader, model, is_save_result=True)

    print('test on {}, the result are: S_lambda = {:0.4f}, F_beta = {:0.4f}, Mae = {:0.4f}'.format(args.test_dataset,
                                                                                                   S_lambda, F_beta,
                                                                                                   Mae))
    logging.info(
        'test on {}, the result are: S_lambda = {:0.4f}, F_beta = {:0.4f}, Mae = {:0.4f}'.format(args.test_dataset,
                                                                                                 S_lambda, F_beta, Mae))
