from .utils.exp import train_net, grid_search
import multiprocessing
import os


def run_exp(exp_dir, **search_params):
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    searches = grid_search(**search_params)

    for i, search in enumerate(searches):
        save_dir = os.path.join(exp_dir, str(i))
        search["save_dir"] = save_dir
        p = multiprocessing.Process(target=train_net, kwargs=search)
        p.start()
        p.join()

