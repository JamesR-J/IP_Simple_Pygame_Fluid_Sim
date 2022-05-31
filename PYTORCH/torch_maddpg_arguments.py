import torch
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser("MARL within particle fluid simulation")

    # environment
    parser.add_argument("--per_episode_max_len", type=int, default=10000, help="maximum episode length")  # 10000
    parser.add_argument("--max_episode", type=int, default=100, help="maximum number of episodes")  # 100

    # core training parameters
    parser.add_argument("--device", default=device, help="torch device")
    parser.add_argument("--learning_start_step", type=int, default=50000, help="learning start steps")  # 50000
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")  # 0.5
    parser.add_argument("--learning_fre", type=int, default=150, help="learning frequency")  # 150
    parser.add_argument("--tao", type=int, default=0.01, help="marl tao")
    parser.add_argument("--lr_a", type=float, default=1e-2, help="learning rate for adam actor")  # 1e-2
    parser.add_argument("--lr_c", type=float, default=1e-2, help="learning rate for adam critic")  # 1e-2
    parser.add_argument("--gamma", type=float, default=0.97, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=1256, help="number of episodes to optimise at the same time")
    parser.add_argument("--memory_size", type=int, default=1e6, help="size of data stored in memory")
    parser.add_argument("--num_units_1", type=int, default=128, help="number of units in the NN")
    parser.add_argument("--num_units_2", type=int, default=64, help="number of units in the NN")

    # checkpointing
    parser.add_argument("--fre4save_model", type=int, default=400, help="episode number for saving the model")  # 400
    parser.add_argument("--start_save_model", type=int, default=400, help="episode number for saving the model")  # 400
    parser.add_argument("--save_dir", type=str, default="torch_maddpg_save", \
                        help="directory in which training state and model are saved")
    parser.add_argument("--old_model_name", type=str, default="load_models/longest_run_besta_model_save/", \
                        help="directory in which training state and model are loaded")

    # evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    return parser.parse_args()
