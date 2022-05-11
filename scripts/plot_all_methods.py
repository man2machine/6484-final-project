import argparse
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["dt", "lstm_dt", "s4_dt"]

def plot(args):
    print("*****")

    env_d4rl_name = args.env_d4rl_name
    log_dir = args.log_dir
    x_key = args.x_key
    y_key = args.y_key
    y_smoothing_win = args.smoothing_window
    sparse = ""
    all_files_per_model = {model: None for model in MODELS}
    
    if args.rtg_sparse_flag == 'True':
        sparse = "_sparse"
    
    for model in MODELS:
        all_files_per_model[model] = glob.glob('dt_runs/' + f'/{model}_{env_d4rl_name}{sparse}_log_22*.csv')
        
    save_fig_path = f'plots/all_models_'+ env_d4rl_name + sparse + "_avg.png"

    ax = plt.gca()
    ax.set_title(f'Averages for All Models on Env: {env_d4rl_name+sparse}')

    name_list = []
    df_list = []
    for model_name, all_files in all_files_per_model.items():
        print(model_name)
        for filename in all_files:
            frame = pd.read_csv(filename, index_col=None, header=0)
            print(filename, frame.shape)
            frame['y_smooth'] = frame[y_key].rolling(window=y_smoothing_win).mean()
            df_list.append(frame)

        df_concat = pd.concat(df_list)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()
        data_std = df_concat_groupby.std()

        data_avg.plot(x=x_key, y='y_smooth', ax=ax, label=model_name)
        ax.fill_between(data_avg[x_key], data_avg['y_smooth'] - data_std['y_smooth'], data_avg['y_smooth'] + data_std['y_smooth'], alpha=0.4)
        print((data_avg['y_smooth'] - data_std['y_smooth']).values[-1])
        print(data_avg['y_smooth'].values[-1], data_std['y_smooth'].values[-1])
        print((data_avg['y_smooth'] + data_std['y_smooth']).values[-1])

    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend(loc='lower right')
    plt.savefig(save_fig_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name', type=str, default='dt')
    parser.add_argument('--env_d4rl_name', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--rtg_sparse_flag', type=str, default='False')
    parser.add_argument('--log_dir', type=str, default='dt_runs/')
    parser.add_argument('--x_key', type=str, default='num_updates')
    parser.add_argument('--y_key', type=str, default='eval_d4rl_score')
    parser.add_argument('--smoothing_window', type=int, default=10)

    args = parser.parse_args()

    plot(args)
