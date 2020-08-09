from .utils.exp import train_net, test_net, grid_search
import multiprocessing
import os


def run_test(model, data_dir='/root/data/vessels/test/images', seg_dir='/root/data/vessels/test/gt',
             test_metrics=(('accuracy', {}), ), save_dir='/root/output/vessels', save_preds=False, **params):

    test_net(model, data_dir=data_dir, seg_dir=seg_dir, test_metrics=test_metrics, save_dir=save_dir,
             save_preds=save_preds, **params)


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




