"""Modified from https://github.com/mlfoundations/open_flamingo"""

import argparse
import copy
import glob
import os
import random
import time

import numpy as np
import torch
import wandb
from mmengine import Config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from mmgpt import create_model_and_transforms
from mmgpt.models.builder import create_toy_model_and_transforms
from mmgpt.datasets import InfiniteSampler, build_dataset
from mmgpt.train.distributed import init_distributed_device, world_info_from_env
from mmgpt.train.train_utils import AverageMeter, get_autocast, get_cast_dtype, get_checkpoint


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_encoder_path", default="ViT-L-14", type=str)
    parser.add_argument("--vision_encoder_pretrained", default="openai", type=str)
    parser.add_argument("--lm_path", default="checkpoints/llama-7b_hf", type=str)
    parser.add_argument(
        "--tokenizer_path",
        default="checkpoints/llama-7b_hf",
        type=str,
        help="path to tokenizer",
    )
    parser.add_argument(
        "--pretrained_path",
        default="checkpoints/OpenFlamingo-9B/checkpoint.pt",
        type=str,
        help="path to pretrained model",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="train-my-gpt4",
        help="used to name saving directory and wandb run",
    )
    parser.add_argument("--use_media_placement_augmentation", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100, help="log loss every n steps")
    # Sum of gradient optimization batch size
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
        default=None,
    )
    parser.add_argument(
        "--delete_previous_checkpoint",
        action="store_true",
        help="delete previous checkpoint when saving new checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        type=str,
        help="constant, linear, or cosine",
    )
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision.",
    )
    # data args
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset_config", type=str, default=None, help="path to dataset config file")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    # Finetune config
    parser.add_argument("--tuning_config", type=str, default=None, help="path to tuning config file")
    # distributed training args
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
    )
    # wandb args
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument(
        "--wandb_project",
        type=str,
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
    )
    parser.add_argument(
        "--save_checkpoints_to_wandb",
        default=False,
        action="store_true",
        help="save checkpoints to wandb",
    )

    args = parser.parse_args()

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if args.rank == 0:
        if not os.path.exists(args.run_name):
            os.makedirs(args.run_name)

    device_id = init_distributed_device(args)

    random_seed(args.seed)

    if args.tuning_config is not None:
        tuning_config = Config.fromfile(args.tuning_config)
    else:
        raise ValueError("tuning_config must be specified")

    model, image_processor, tokenizer = create_model_and_transforms(
        model_name="open_flamingo",
        clip_vision_encoder_path=args.vision_encoder_path,
        clip_vision_encoder_pretrained=args.vision_encoder_pretrained,
        lang_encoder_path=args.lm_path,
        tokenizer_path=args.tokenizer_path if args.tokenizer_path else args.lm_path,
        use_media_placement_augmentation=args.use_media_placement_augmentation,
        pretrained_model_path=args.pretrained_path,
        tuning_config=tuning_config.tuning_config,
    )

    if args.dataset_config is not None:
        dataset_config = Config.fromfile(args.dataset_config)
    else:
        raise ValueError("dataset_config must be specified")

    dataset = build_dataset(
        dataset_config=dataset_config.visual_datasets,
        vis_processor=image_processor,
        tokenizer=tokenizer,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=DistributedSampler(dataset, shuffle=True, drop_last=True),
        collate_fn=dataset.collater,
    )

    # build language dataset and dataloader for multi-modality training
    if dataset_config.get('language_datasets') is not None and len(dataset_config.language_datasets) > 0:
        lang_dataset = build_dataset(
            dataset_config=dataset_config.language_datasets,
            tokenizer=tokenizer,
        )
        lang_dataloader = DataLoader(
            lang_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
            sampler=InfiniteSampler(lang_dataset, shuffle=True),
            collate_fn=lang_dataset.collater,
        )
        lang_dataloader = iter(lang_dataloader)
    else:
        lang_dataloader = None

    random_seed(args.seed, args.rank)

    print(f"Start running training on rank {args.rank}.")

    if args.rank == 0 and args.report_to_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
        )

    device_id = args.rank % torch.cuda.device_count()
    model = model.to(device_id)

    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        for n, p in model.named_parameters():
            # if p.requires_grad:
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": params_with_wd, "weight_decay": args.weight_decay},
            {"params": params_without_wd, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), lr=args.learning_rate)

    total_training_steps = len(train_dataloader) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps // args.gradient_accumulation_steps,
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    # check if a checkpoint exists for this run
    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        checkpoint_list = glob.glob(f"{args.run_name}/checkpoint_*.pt")
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}.")

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None:
        if args.rank == 0:
            print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        resume_from_epoch = checkpoint["epoch"] + 1

    ddp_model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        train_one_epoch(
            args=args,
            model=ddp_model,
            epoch=epoch,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            language_dataloader=lang_dataloader,
            device_id=device_id,
            wandb=wandb,
        )

        if args.rank == 0:
            if not os.path.exists(args.run_name):
                os.makedirs(args.run_name)

            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "tuning_config": tuning_config,
            }

            print(f"Saving checkpoint to {args.run_name}/checkpoint_{epoch}.pt")
            torch.save(checkpoint_dict, f"{args.run_name}/checkpoint_{epoch}.pt")
            if args.report_to_wandb and args.save_checkpoints_to_wandb:
                wandb.save(f"{args.run_name}/checkpoint_{epoch}.pt")

            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(f"{args.run_name}/checkpoint_{epoch-1}.pt")
    if args.rank == 0:
        torch.save(
            {"model_state_dict": get_checkpoint(ddp_model.module), "tuning_config": tuning_config},
            f"{args.run_name}/final_weights.pt",
        )
        if args.report_to_wandb and args.save_checkpoints_to_wandb:
            wandb.save(f"{args.run_name}/final_weights.pt")


def train_one_epoch(
    args,
    model,
    epoch,
    train_dataloader,
    language_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb,
):
    num_batches_per_epoch = len(train_dataloader)

    total_training_steps = num_batches_per_epoch * args.num_epochs

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    model.train()

    # setup logging
    step_time_m = AverageMeter()  # time for one optimizer step (> 1 batch if using gradient accum)
    data_time_m = (
        AverageMeter()
    )  # avg time to load one batch of both C4 AND laion (= 1 batch regardless of gradient accum)
    end = time.time()

    # loop through dataloader
    for num_steps, batch in tqdm(
        enumerate(train_dataloader),
        disable=args.rank != 0,
        total=total_training_steps,
        initial=(epoch * num_batches_per_epoch),
    ):
        data_time_m.update(time.time() - end)

        global_step = num_steps + epoch * num_batches_per_epoch

        #### VISION FORWARD PASS ####
        images = batch["image"].to(device_id, dtype=cast_dtype, non_blocking=True).unsqueeze(1).unsqueeze(1)
        input_ids = batch["input_ids"].to(device_id, dtype=cast_dtype, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
        labels = batch["labels"].to(device_id, dtype=cast_dtype, non_blocking=True)

        with autocast():
            loss_batch = model(
                vision_x=images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )[0]
        loss = loss_batch / args.gradient_accumulation_steps
        loss_vision = loss  # for logging

        #### BACKWARD PASS ####
        loss.backward()

        #### LANGUAGE FORWARD PASS ####
        if language_dataloader is not None:
            batch_lang = next(language_dataloader)
            lang_input_ids = batch_lang["input_ids"].to(device_id, dtype=cast_dtype, non_blocking=True)
            lang_attention_mask = batch_lang["attention_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
            lang_labels = batch_lang["labels"].to(device_id, dtype=cast_dtype, non_blocking=True)

            with autocast():
                lang_loss_batch = model(
                    vision_x=None,
                    lang_x=lang_input_ids,
                    attention_mask=lang_attention_mask,
                    labels=lang_labels,
                )[0]
            lang_loss = lang_loss_batch / args.gradient_accumulation_steps
            #### BACKWARD PASS ####
            lang_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step optimizer and log
        if (((num_steps + 1) % args.gradient_accumulation_steps) == 0) or (num_steps == num_batches_per_epoch - 1):
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # step time and reset end outside of rank 0
            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb:
                # compute within rank 0
                samples_per_second = (
                    args.gradient_accumulation_steps * args.batch_size * args.world_size / step_time_m.val
                )
                samples_per_second_per_gpu = args.gradient_accumulation_steps * args.batch_size / step_time_m.val

                wandb.log(
                    {
                        "data_time": data_time_m.avg,
                        "step_time": step_time_m.avg,
                        "samples_per_second": samples_per_second,
                        "samples_per_second_per_gpu": samples_per_second_per_gpu,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    commit=False,
                )
                step_time_m.reset()
                data_time_m.reset()

                loss_log = {
                    "loss": loss.item(),
                    "loss_vision": loss_vision.item(),
                    "global_step": global_step,
                }
                if language_dataloader is not None:
                    loss_log["loss_lang"] = lang_loss.item()

                wandb.log(loss_log, commit=True)

        # Log loss to console
        if ((num_steps + 1) % args.logging_steps == 0) and args.rank == 0:
            print(
                f"Step {num_steps+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. Loss: {loss.item():.3f}"
            )


if __name__ == "__main__":
    main()
