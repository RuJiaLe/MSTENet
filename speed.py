import numpy as np
import torch
import time
from model.model import Model
from options import args
from tqdm import tqdm


def computeTime(model, device):

    model = model.to(device)
    
    model.eval()

    time_spent = []

    for idx in tqdm(range(110)):

        inputs = torch.randn(args.clip_len, 3, args.size, args.size)

        inputs = inputs.to(device)

        start_time = time.time()

        with torch.no_grad():
            _ = model(inputs)
        
        if device == 'cuda':
            torch.cuda.synchronize()

        if idx > 10:
            time_spent.append((time.time() - start_time) / inputs.size(0))
    
    return time_spent


if __name__=="__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model()

    time_speed = computeTime(model, device)

    print('Avg execution time (s): %.4f, FPS:%d' % (np.mean(time_speed), 1 // np.mean(time_speed)))



