import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot(args):
    
    model_name = args.model_name
    env_d4rl_name = args.env_d4rl_name
    log_dir = args.log_dir
    x_key = args.x_key
    y_key = args.y_key
    y_smoothing_win = args.smoothing_window
    plot_avg = args.plot_avg
    sparse = ""
    all_files = glob.glob(log_dir + f'/{model_name}_{env_d4rl_name}_log_22*.csv')
    if args.rtg_sparse_flag == 'True':
        sparse = "_sparse"
        all_files = glob.glob(log_dir + f'/{model_name}_{env_d4rl_name}_sparse_log_22*.csv')
    
    if plot_avg:
        save_fig_path = f'plots/{model_name}_'+ env_d4rl_name + sparse + "_avg.png"
    else:
        save_fig_path = f'plots/{model_name}_'+ env_d4rl_name + sparse + ".png"
        max_save_fig_path = f'plots/max_{model_name}_'+ env_d4rl_name + sparse + ".png"

    ax = plt.gca()
    ax.set_title(f'Model: {model_name}, Env: {env_d4rl_name+sparse}')

    if plot_avg:
        name_list = []
        df_list = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            frame['y_smooth'] = frame[y_key].rolling(window=y_smoothing_win).mean()
            df_list.append(frame)

        df_concat = pd.concat(df_list)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()

        data_avg.plot(x=x_key, y='y_smooth', ax=ax)

        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.legend(['avg of all runs'], loc='lower right')
        plt.savefig(save_fig_path)

    else:
        name_list = []
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            frame['y_smooth'] = frame[y_key].rolling(window=y_smoothing_win).mean()
            frame.plot(x=x_key, y='y_smooth', ax=ax)
            name_list.append(filename.split('/')[-1])

        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.legend(name_list, loc='lower right')
        plt.savefig(save_fig_path)
    
    plt.figure()
    ax = plt.gca()
    ax.set_title(f'Maximum Model: {model_name}, Env: {env_d4rl_name+sparse}')
    name_list = []
    for filename in all_files:
        frame = pd.read_csv(filename, index_col=None, header=0)
#         print(filename, frame.shape)
#         print(np.maximum.accumulate(frame[y_key]))
        frame['y_max'] = frame[y_key].cummax()
#         print(frame['y_max'])
        frame.plot(x=x_key, y='y_max', ax=ax)
        name_list.append(filename.split('/')[-1])

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend(name_list, loc='lower right')
    plt.savefig(max_save_fig_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dt')
    parser.add_argument('--env_d4rl_name', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--rtg_sparse_flag', type=str, default='False')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')
    parser.add_argument('--x_key', type=str, default='num_updates')
    parser.add_argument('--y_key', type=str, default='eval_d4rl_score')
    parser.add_argument('--smoothing_window', type=int, default=1)
    parser.add_argument("--plot_avg", action="store_true", default=False,
                    help="plot avg of all logs else plot separately")

    args = parser.parse_args()

    plot(args)
