import sys
import torch
from tqdm import tqdm as tqdm
import numpy as np
from .meter import AverageValueMeter
import time

class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            if metric != "inf_time":
                metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.name : AverageValueMeter() for metric in self.metrics if metric != "inf_time"}
        metrics_meters.update({"inf_time": []} if "inf_time" in self.metrics else {})

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, (y, wt) in iterator:
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
                        metrics_meters[metric_fn] = metrics_meters[metric_fn] + [inf_time]
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

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, wt):
        self.optimizer.zero_grad()
        start = time.time()
        prediction = self.model.forward(x)
        end = time.time()
        inf_time = end - start
        loss = self.loss(prediction, y)
        loss = torch.sum(loss * wt)/torch.sum(wt)
        loss.backward()
        self.optimizer.step()
        return loss, prediction, inf_time


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, wt):
        with torch.no_grad():
            start = time.time()
            prediction = self.model.forward(x)
            end = time.time()
            inf_time = end - start
            loss = self.loss(prediction, y)
            loss = torch.sum(loss * wt) / torch.sum(wt)
        return loss, prediction, inf_time
