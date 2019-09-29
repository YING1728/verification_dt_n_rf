"""
A script to reconstruct the decision table for the ACAS XU controller
(cf. Julien et al. 2016). The script samples data from a DNN in nnet fromat
and outputs the reconstructed table in csv format. The nnet files are taken
from the Reluplex github by Guy Katz

https://github.com/guykatzz/ReluplexCav2017/tree/master/nnet


Sample usage:
    python reconstruct_table.py --nnet-filepath
                    nnet/ACASXU_run2a_1_1_batch_2000.nnet
"""
import os
from os.path import dirname, join as path_join

import argparse

from util import *

import torch
from torch.distributions import Uniform

import numpy as np

import sys
import logging

FORMAT = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)

TABLES_DIR = "tables"


def sample_input(num_samples, minimum, maximum):
    m = Uniform(torch.tensor(minimum), torch.tensor(maximum))
    return m.sample(sample_shape=(num_samples,))


def generate_table_entries(args, soft_labels=False):
    nnet_file = open(args.nnet_filepath)
    acas_net = AcasNetworkParser(nnet_file)
    net = acas_net.to_torch()

    num_samples = args.num_samples
    input_tensor = sample_input(
        args.num_samples,
        acas_net.inp_mins,
        acas_net.inp_maxs)

    if soft_labels:
        output_tensor =  net.forward(input_tensor)
    else:
        output_tensor_1 = net.forward(input_tensor).min(1)[1].view(num_samples, 1) # lowest score means the best action.
        output_tensor_2 = net.forward(input_tensor).max(1)[1].view(num_samples,1) # highest score means the worst action.
    training_data = torch.cat((input_tensor.float(), output_tensor_1.float(), output_tensor_2.float()), 1)
    np.savetxt(args.output_filepath,
               training_data.detach().numpy(), delimiter=",")
    file_size_in_MB = os.path.getsize(args.output_filepath) >> 20
    num_floats = training_data.numel()
    return num_floats, file_size_in_MB


def create_directory_if_not_exists(args):
    path = path_join(dirname(__file__), TABLES_DIR)
    os.makedirs(path, exist_ok=True)
    if not args.output_filepath:
        nnet_filepath_prefix = (
            args.nnet_filepath.split('/')[-1]).split('.')[0]
        csv_filename = "table_" + nnet_filepath_prefix + ".csv"
        args.output_filepath = path_join(path, csv_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-samples",
        help="Number of samples.",
        type=int,
        default=10000
    )
    parser.add_argument(
        "--soft-labels",
        help="Use soft labels",
        action="store_true"
    )
    parser.add_argument(
        "--nnet-filepath",
        help="Path to an nnet file.",
        required=True
    )
    parser.add_argument(
        "--output-filepath",
        help="Path to the output csv file representing the table.",
        default=None
    )


    args = parser.parse_args()
    logging.info("Parsing Network `%s`." % args.nnet_filepath)

    create_directory_if_not_exists(args)

    logging.info("Sampling %s table entries..." % args.num_samples)
    num_floats, file_size_in_MB = generate_table_entries(args, args.soft_labels)

    logging.info("Wrote output to %s [%s Floats, %s MB]" %
                 (args.output_filepath, num_floats, file_size_in_MB))


if __name__ == "__main__":
    main()
