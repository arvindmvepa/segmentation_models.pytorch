from .utils.exp import train_net, val_net, test_net, grid_search
import multiprocessing
from glob import glob
import json
import os


def run_test(model, **params):
    test_net(model, **params)


def run_val_exp(model_bname, exp_dir, **params):
    job_dirs = list(glob(exp_dir))
    for job_dir in job_dirs:
        params_file = os.path.join(job_dir, "params.json")
        with open(params_file) as json_file:
            job_params = json.load(json_file)
        job_params.update(params)
        job_params['model_path'] = os.path.join(job_dir, model_bname)
        print(job_params)
        p = multiprocessing.Process(target=val_net, kwargs=job_params)
        p.start()
        p.join()


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




