import glob
import logging
import math
import os
import random
from typing import *

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup


from utils import AverageMeter, reduce_mean


class Trainer:
    def __init__(self, hparams, tokenizer, loaders, model, resultwriter):
        self.hparams = hparams
        self.rank = self.hparams.rank
        self.nprocs = torch.cuda.device_count()
        self.scaler = torch.cuda.amp.GradScaler() if self.hparams.amp else None
        if self.hparams.distributed:
            assert torch.cuda.is_available()
            self.device = f"cuda:{self.rank}"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.model = model.to(self.device, non_blocking=True)
        if self.hparams.distributed:
            self.model = DDP(self.model, device_ids=[self.rank])

        self.tokenizer = tokenizer
        self.pad_idx = self.tokenizer.vocab["<pad>"]

        # dataloader and distributed sampler
        self.train_loader, self.valid_loader, self.test_loader = loaders
        self.train_sampler = self.train_loader.sampler

        # optimizer, scheduler
        self.optimizer, self.scheduler = self.configure_optimizers()

        # model saving options
        self.global_step = 0
        self.eval_step = int(self.step_total * hparams.eval_ratio)
        if self.rank in [-1, 0]:
            self.version = 0
            while True:
                self.save_path = os.path.join(
                    hparams.ckpt_path, f"version-{self.version}"
                )
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                    break
                else:
                    self.version += 1
            self.summarywriter = SummaryWriter(self.save_path)
            self.global_val_loss = float("inf")
            self.log_step = hparams.log_step
            logging.basicConfig(
                filename=os.path.join(self.save_path, "experiment.log"),
                level=logging.INFO,
                format="%(asctime)s > %(message)s",
            )
            with open(
                os.path.join(self.save_path, "hparams.yaml"), "w", encoding="utf8"
            ) as outfile:
                yaml.dump(
                    hparams, outfile, default_flow_style=False, allow_unicode=True
                )

            # experiment logging options
            self.best_result = {"version": self.version}

            logging.info(
                f"[SCHEDULER] Total_step: {self.step_total} | Warmup step: {self.warmup_steps}"
            )

    def configure_optimizers(self):
        # optimizer
        decay_parameters = self.get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)

        # lr warmup scheduler
        self.step_total = len(self.train_loader) * self.hparams.epoch
        self.warmup_steps = math.ceil(self.step_total * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.step_total,
        )

        return optimizer, scheduler

    def get_parameter_names(self, model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result

    def save_checkpoint(self, epoch: int, val_loss: float, model: nn.Module) -> None:
        logging.info(
            f"      Val loss decreased ({self.global_val_loss:.4f} → {val_loss:.4f}). Saving model ..."
        )
        new_path = os.path.join(
            self.save_path, f"best_model_step_{self.global_step}_loss_{val_loss:.4f}.pt"
        )

        for filename in glob.glob(os.path.join(self.save_path, "*.pt")):
            os.remove(filename)  # remove old checkpoint
        torch.save(model.state_dict(), new_path)
        self.global_val_loss = val_loss

    def fit(self) -> dict:
        for epoch in tqdm(
            range(self.hparams.epoch), desc="epoch", disable=self.rank not in [-1, 0]
        ):
            if self.hparams.distributed:
                self.train_sampler.set_epoch(epoch)

            self._train_epoch(epoch)

        if self.rank in [-1, 0]:
            self.summarywriter.close()
        return self.best_result if self.rank in [-1, 0] else None

    def _train_epoch(self, epoch: int) -> None:
        train_loss = AverageMeter()

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader),
            desc="train_steps",
            total=len(self.train_loader),
            disable=self.rank not in [-1, 0],
        ):
            input_ids = batch["input_ids"]
            attention_mask = input_ids.ne(self.pad_idx).float()
            dec_input_ids = batch["decoder_input_ids"]
            dec_attention_mask = dec_input_ids.ne(self.pad_idx).float()
            labels = batch["labels"]

            # load to machine
            input_ids = input_ids.to(self.device, non_blocking=True)
            attention_mask = attention_mask.to(self.device, non_blocking=True)
            dec_input_ids = dec_input_ids.to(self.device, non_blocking=True)
            dec_attention_mask = dec_attention_mask.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # compute loss
            self.optimizer.zero_grad()
            if self.hparams.amp:
                with torch.cuda.amp.autocast():
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = output.loss
            else:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = output.loss

            # reduce and update
            if self.hparams.distributed:
                dist.barrier()
                loss = reduce_mean(loss, self.nprocs)

            if self.hparams.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scheduler.step()
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            train_loss.update(loss.item())

            # validate and logging
            self.global_step += 1
            if self.global_step % self.eval_step == 0:
                val_loss = self.validate(epoch)
                if self.rank in [-1, 0]:
                    self.summarywriter.add_scalars(
                        "loss/step", {"val": val_loss}, self.global_step
                    )
                    logging.info(
                        f"[VAL] global step: {self.global_step} | val loss: {val_loss:.3f}"
                    )
                    if val_loss < self.global_val_loss:
                        self.save_checkpoint(epoch, val_loss, self.model)

            # train logging
            if self.rank in [-1, 0]:
                if self.global_step % self.log_step == 0:
                    logging.info(
                        f"[TRN] Version: {self.version} | Epoch: {epoch} | Global step: {self.global_step} | Train loss: {loss.item():.3f} | LR: {self.optimizer.param_groups[0]['lr']:.5f}"
                    )
                    self.summarywriter.add_scalars(
                        "loss/step", {"train": train_loss.avg}, self.global_step
                    )
                    self.summarywriter.add_scalars(
                        "lr",
                        {"lr": self.optimizer.param_groups[0]["lr"]},
                        self.global_step,
                    )

    def validate(self, epoch: int) -> float:
        val_loss = AverageMeter()

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.valid_loader),
                desc="valid_steps",
                total=len(self.valid_loader),
                disable=self.rank not in [-1, 0],
            ):
                input_ids = batch["input_ids"]
                attention_mask = input_ids.ne(self.pad_idx).float()
                dec_input_ids = batch["decoder_input_ids"]
                dec_attention_mask = dec_input_ids.ne(self.pad_idx).float()
                labels = batch["labels"]

                # load to machine
                input_ids = input_ids.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)
                dec_input_ids = dec_input_ids.to(self.device, non_blocking=True)
                dec_attention_mask = dec_attention_mask.to(
                    self.device, non_blocking=True
                )
                labels = labels.to(self.device, non_blocking=True)

                # compute loss
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input_ids,
                    decoder_attention_mask=dec_attention_mask,
                    labels=labels,
                )
                loss = output.loss

                # reduce
                if self.hparams.distributed:
                    dist.barrier()
                    loss = reduce_mean(loss, self.nprocs)
                val_loss.update(loss.item())

        return val_loss.avg

    def test(self, state_dict) -> dict:
        test_loss = AverageMeter()

        self.model.load_state_dict(state_dict)
        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.test_loader),
                desc="tst_steps",
                total=len(self.test_loader),
            ):
                input_ids = batch["input_ids"]
                attention_mask = input_ids.ne(self.pad_idx).float()
                dec_input_ids = batch["decoder_input_ids"]
                dec_attention_mask = dec_input_ids.ne(self.pad_idx).float()
                labels = batch["labels"]

                # load to machine
                input_ids = input_ids.to(self.device, non_blocking=True)
                attention_mask = attention_mask.to(self.device, non_blocking=True)
                dec_input_ids = dec_input_ids.to(self.device, non_blocking=True)
                dec_attention_mask = dec_attention_mask.to(
                    self.device, non_blocking=True
                )
                labels = labels.to(self.device, non_blocking=True)

                # compute loss
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=dec_input_ids,
                    decoder_attention_mask=dec_attention_mask,
                    labels=labels,
                )
                loss = output.loss
                test_loss.update(loss.item())

        logging.info(f"[TST] Test Loss: {test_loss.avg:.4f}")

        return {"test_loss": test_loss.avg}
