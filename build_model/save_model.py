

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm





class SaveModel():
    def __init__(self,model,save_dir=''):
        self.model=model
        if not save_dir:
            save_dir=os.makedirs('weights',exist_ok=True)
        self.save_dir=save_dir
        self.last_weight_path=os.path.join(save_dir,'last.pt')
        self.best_weight_path=os.path.join(save_dir,'best.pt')

        self.best_fitness, self.start_epoch = 0.0, 0

    def __call__(self, epoch,):

        # Save model
        if (not nosave) or (final_epoch and not evolve):  # if save
            ckpt = {
                'epoch': epoch,
                'best_fitness': self.best_fitness,
                'model': deepcopy(self.model),
                'ema': deepcopy(ema.ema).half(),
                'updates': ema.updates,
                'optimizer': optimizer.state_dict(),
                'opt': vars(opt),
                # 'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                'date': datetime.now().isoformat()}

            # todo  Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            if opt.save_period > 0 and epoch % opt.save_period == 0:
                torch.save(ckpt, w / f'epoch{epoch}.pt')
                logger.log_model(w / f'epoch{epoch}.pt')
            del ckpt
            # callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        for f in last, best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                        mask_downsample_ratio=mask_ratio,
                        overlap=overlap)  # val best model with plots
                    if is_coco:
                        # callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
                        metrics_dict = dict(zip(KEYS, list(mloss) + list(results) + lr))
                        logger.log_metrics(metrics_dict, epoch)

        # callbacks.run('on_train_end', last, best, epoch, results)
        # on train end callback using genericLogger
        logger.log_metrics(dict(zip(KEYS[4:16], results)), epochs)
        if not opt.evolve:
            logger.log_model(best, epoch)
        if plots:
            plot_results_with_masks(file=save_dir / 'results.csv')  # save results.png
            files = ['results.png', 'confusion_matrix.png', *(f'{x}_curve.png' for x in ('F1', 'PR', 'P', 'R'))]
            files = [(save_dir / f) for f in files if (save_dir / f).exists()]  # filter
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
            logger.log_images(files, 'Results', epoch + 1)
            logger.log_images(sorted(save_dir.glob('val*.jpg')), 'Validation', epoch + 1)
    torch.cuda.empty_cache()
    return results

    def fitness(self,x):
        # Model fitness as a weighted combination of metrics
        w = [0.0, 0.0, 0.1, 0.9, 0.0, 0.0, 0.1, 0.9]
        return (x[:, :8] * w).sum(1)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
