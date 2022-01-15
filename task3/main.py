import argparse
import numpy as np
import os
import csv
import utilities
import torch
import matplotlib.pyplot as plt

from dataset import ValidationDataset, DebugDataset
from video_processing import DynamicVideoProcessing
from models import LoDE

from config import *

if __name__ == '__main__':
    print("current device: {}".format('cuda' if torch.cuda.is_available() else 'cpu'))

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', help='Path to the database directory.')
    parser.add_argument('--draw', default=True, action='store_true', help='Output visual results in ./results')
    parser.add_argument('--predict_on_private', dest='predict_on_private', action='store_true', default=False)
    parser.add_argument('--test', dest='test', action='store_true', default=True)

    args = parser.parse_args()

    if args.test:
        phase = 'validation_test'
        object_set = [str(i + 1) for i in range(9)]
        output_path = 'results'
        if not os.path.exists(output_path): os.makedirs(output_path)

        dataset = ValidationDataset(args.data_path)
        video_processing = DynamicVideoProcessing(args, output_path)

        print(f'Executing on {object_set} containers...')
        lode = LoDE(args, output_path, phase, video_processing, dataset=dataset)
        output_path = 'results'
        tag = "v1.0"
        with open('{}/estimation_{}_{}.csv'.format(output_path, tag, phase), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['fileName', 'height[mm]', 'width[mm]', 'capacity[mL]', 'frame', 'capacity_diff[mL]', 'score'])

        for args.object in object_set:
            lode.run(tag=tag)

        # calculate final capacities
        scores, cscores = utilities.calc_final_estimation(dataset, average_training_set, phase, [tag])
        print("scores: {:.3f}".format(np.mean(scores)))

        for cid in sorted(list(cscores.keys())):
            print("{}: {:.3f}".format(cid, np.mean(cscores[cid])))

    elif args.predict_on_private:
        phase = 'private_test'
        object_set = ['13', '14', '15']
        pass
    else:
        phase = 'public_test'
        object_set = ['10', '11', '12']
        pass

    print('Completed!')
