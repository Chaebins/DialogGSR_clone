import functools
import logging
import os

import torch
from torch.optim import Adam
from tqdm import tqdm
from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup

from torch.cuda.amp import autocast, GradScaler

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, model, dataloader, num_train_steps, writer=None, step_callback=None):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.num_train_steps = num_train_steps
        self.step_callback = step_callback

        self.optimizer = self._create_optimizer(model)
        self.scheduler = self._create_scheduler(self.optimizer)

    def train(self):
        model = self.model
        dataloader = self.dataloader

        scaler = GradScaler()


        epoch = 0
        global_step = 0
        tr_loss = 0.0

        model.train()
        with tqdm(total=self.num_train_steps, disable=self.args.local_rank not in (-1, 0)) as pbar:
            while True:
                for step, batch in enumerate(dataloader):
                    model.train()
                    inputs = dict()
                    inputs = {k: v.to(self.args.device) for k, v in self._create_model_arguments(batch).items() \
                        if k not in ["episode_id", "turn_id"]}                            
                    inputs.pop("episode_id")
                    inputs.pop("turn_id")
                    
                    if self.args.fp16:
                        with autocast():
                            outputs = model(**inputs)
                            loss = outputs[0]
                    else:
                        outputs = model(**inputs)
                        loss = outputs[0]

                    if type(loss) == dict:
                        loss = loss['total_loss']

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps

                    if self.args.fp16:
                        scaler.scale(loss).backward() # fp16
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    # Gradient Accumulation
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        if self.args.fp16:
                            scaler.unscale_(self.optimizer) # fp16
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        if self.args.fp16:
                            scaler.step(self.optimizer) # fp16
                        else:
                            self.optimizer.step()

                        if self.args.use_contrastive:
                            if hasattr(model, "module"):
                                model.module.batch_scale.data = \
                                    torch.clamp(model.module.batch_scale.data, 0, 4.6052)
                                model.module.self_scale.data = \
                                    torch.clamp(model.module.self_scale.data, 0, 4.6052)
                            else:
                                model.batch_scale.data = torch.clamp(model.batch_scale.data, 0, 4.6052)
                                model.self_scale.data = torch.clamp(model.self_scale.data, 0, 4.6052)

                        model.zero_grad()
                        if self.args.fp16:
                            scaler.update()

                        self.scheduler.step()
                        pbar.set_description("epoch: %d loss: %.7f" % (epoch, loss.item()))
                        pbar.update()
                        global_step += 1

                        if self.step_callback is not None:
                            self.step_callback(model, global_step)

                        if (
                            self.args.local_rank in (-1, 0)
                            and self.args.output_dir
                            and self.args.save_steps > 0
                            and global_step % self.args.save_steps == 0
                        ):
                            output_dir = os.path.join(self.args.output_dir, "checkpoint-{}".format(global_step))

                            if hasattr(model, "module"):
                                model.save_pretrained(output_dir)
                            else:
                                model.save_pretrained(output_dir)

                        if global_step == self.num_train_steps:
                            break
                if global_step == self.num_train_steps:
                    break
                epoch += 1

        logger.info("global_step = %s, average loss = %s", global_step, tr_loss / global_step)
        return model, global_step, tr_loss / global_step

    def _create_optimizer(self, model):
        model_params = [(n, p) for n, p in model.named_parameters()]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer = AdamW(
            [
                {
                    "params": [p for n, p in model_params
                            if not any(nd in n for nd in no_decay)
                            and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },

            ],
            eps=self.args.adam_epsilon
        )
        return optimizer

    def _create_scheduler(self, optimizer, warmup_steps=False):
        warmup_steps = int(self.num_train_steps * 0.06)
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, self.num_train_steps)

    def _create_model_arguments(self, batch):
        return batch