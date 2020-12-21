import sys
from tqdm import tqdm as tqdm
import numpy as np
import os
from .meter import AverageValueMeter
from .train import ValidEpoch


class TestEpoch(ValidEpoch):

    def __init__(self, save_preds_dir=None, **kwargs):
        super(TestEpoch, self).__init__(**kwargs)
        self.save_preds_dir = save_preds_dir

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.name : AverageValueMeter() for metric in self.metrics if metric != "inf_time"}
        metrics_meters.update({"inf_time": 0.0} if "inf_time" in self.metrics else {})

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for i, (x, (y, wt)) in enumerate(iterator):
                x, y, wt = x.to(self.device), y.to(self.device), wt.to(self.device)
                loss, y_pred, inf_time = self.batch_update(x, y, wt)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    if metric_fn == "inf_time":
                        metrics_meters[metric_fn] = metrics_meters[metric_fn] + inf_time
                    else:
                        metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                        metrics_meters[metric_fn.name].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items() if k != 'inf_time'}
                if 'inf_time' in metrics_meters:
                    metrics_logs.update({'inf_time': np.mean(metrics_meters['inf_time'])})
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

                # save predictions
                if self.save_preds_dir:
                    test_files = dataloader.dataset.ids
                    test_file = test_files[i]
                    y_pred_values = y_pred.cpu().detach().numpy()
                    np.save(os.path.join(self.save_preds_dir, test_file), y_pred_values)

        return logs
