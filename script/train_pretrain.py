import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import time
import math
import glob
import warnings
import json
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from typing import Optional
from my_tokenizers.hf_math_tokenizer import HFMathTokenizer
from models.model_minimind import MiniMindConfig, MiniMindForCausalLM, MiniMindForRegression
from model_ultils.pretrain_dataset import PretrainDataset, MSEPretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb,start_step, iter_per_epoch, this_run_start_time, cumulative_train_time):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        if step < start_step:
            continue  # 跳过已训练的 steps
        # 🔹 动态解包 batch（causal: (X,Y,mask); mse: (input_ids, targets)）
        if args.loss_type == "causal":
            X, Y, loss_mask = batch
            X = X.to(args.device)
            Y = Y.to(args.device)
            loss_mask = loss_mask.to(args.device)
            profiling_input = X
        else:  # "mse"
            input_ids, targets = batch
            input_ids = input_ids.to(args.device)      # [B, L]
            targets = targets.to(args.device).squeeze(-1)  # [B, 1] → [B]
            profiling_input = input_ids  # ✅ MSE 用 input_ids 做 profiling


        # ✅ 使用传入的 iter_per_epoch 计算全局 step
        global_step = epoch * iter_per_epoch + step
        total_steps = args.epochs * iter_per_epoch
        lr = get_lr(global_step, total_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        profiling = (step % args.log_interval == 0) and (not ddp or dist.get_rank() == 0)
        
        # 🔹 前向 + 损失计算（分模式）
        with ctx:
            if args.loss_type == "causal":
                # ✅ 确保 model 是 MiniMindForCausalLM → 接受 input_ids, 返回 .logits
                res = model(input_ids=X, profiling=profiling)  # ← 明确 keyword
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()
                if hasattr(res, 'aux_loss') and res.aux_loss is not None:
                    loss = loss + res.aux_loss

            else:  # "mse"
                # ✅ 确保 model 是 MiniMindForRegression → 接受 input_ids + labels
                # 注意：MSEPretrainDataset 返回 targets.shape = [B, 1] → 需 squeeze to [B]
                # print(f"[DEBUG RANK{dist.get_rank()}] targets shape: {targets.shape}, dtype: {targets.dtype}, device: {targets.device}")
                # print(f"[DEBUG] targets[:3]: {targets[:3]}")
                targets_squeezed = targets.squeeze(-1)  # [B, 1] → [B]
                # print(f"[DEBUG] targets_squeezed shape: {targets_squeezed.shape}")
                res = model(
                    input_ids=input_ids,
                    labels=targets_squeezed,  # ← 必须是 [B]
                    profiling=profiling
                )
                # print(f"[DEBUG] res.loss = {res.loss}, type={type(res.loss)}")
                loss = res.loss  # ← RegressionOutput.loss
                # 若启用 MOE，res 中可能有 aux_loss → 已在 MiniMindForRegression 中加过了

            loss = loss / args.accumulation_steps
            
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 or step == iter_per_epoch - 1:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.6f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})
            # === 📝 新增：本地持久化 train loss 日志（CSV格式）===
            if not ddp or dist.get_rank() == 0:
                global_step = epoch * iter_per_epoch + step
                loss_val = loss.item() * args.accumulation_steps
                lr_val = optimizer.param_groups[-1]['lr']
                
                loss_log_path = os.path.join(args.save_dir, "train_loss.csv")
                header_needed = not os.path.exists(loss_log_path)
                with open(loss_log_path, "a") as f_loss:
                    if header_needed:
                        f_loss.write("global_step,epoch,step,loss,lr\n")
                    f_loss.write(f"{global_step},{epoch + 1},{step},{loss_val:.8f},{lr_val:.10f}\n")
        #     # === 📝 新增结束 ===
        # if ((step + 1) % args.save_interval == 0 or step == iter_per_epoch - 1) and (not ddp or dist.get_rank() == 0):
        #     model.eval()

        #     # 🔥 动态构造 profiling 输入
        #     if args.loss_type == "causal":
        #         profiling_input = X
        #     else:  # mse
        #         profiling_input = input_ids

        #     # 🔥 新增：强制做一次带 profiling 的前向（使用当前 batch）
        #     with torch.no_grad(), ctx:
        #         res_for_log = model(profiling_input, profiling=True)  # 注意：X 是当前 batch
        #     timestamp = time.strftime("%Y%m%d-%H%M%S")
        #     ckp_dir = os.path.join(args.save_dir, f"checkpoint-epoch{epoch+1}-step{step+1}-{timestamp}")
        #     os.makedirs(ckp_dir, exist_ok=True)

        #     timestamp = time.strftime("%Y%m%d-%H%M%S")
        #     ckp_dir = os.path.join(args.save_dir, f"checkpoint-epoch{epoch+1}-step{step+1}-{timestamp}")
        #     os.makedirs(ckp_dir, exist_ok=True)

        #     actual_model = model.module if isinstance(model, DistributedDataParallel) else model
        #     actual_model.save_pretrained(ckp_dir, safe_serialization=False)

        #     current_this_run_elapsed = time.time() - this_run_start_time
        #     current_total_time = cumulative_train_time + current_this_run_elapsed
        #     # === 新增：保存训练状态 ===
        #     checkpoint = {
        #         'epoch': epoch,
        #         'step': step + 1,
        #         'model_state_dict': actual_model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        #         'lr': optimizer.param_groups[0]['lr'],
        #         'args_dict': vars(args),            # 安全！
        #         'lm_config_dict': lm_config.to_dict() if hasattr(lm_config, 'to_dict') else lm_config.__dict__,
        #         'cumulative_train_time': current_total_time,
        #     }
        #     torch.save(checkpoint, os.path.join(ckp_dir, 'trainer_state.pth'))
        #     # === 🌟 新增：保存 profiling 日志 + 耗时摘要 🌟 ===
        #     if hasattr(res_for_log, 'profiling_logs') and res_for_log.profiling_logs:
        #         log_dir = os.path.join(ckp_dir, "log")
        #         os.makedirs(log_dir, exist_ok=True)  # 这行一定会执行！
        
        #         # 保存原始日志
        #         with open(os.path.join(log_dir, "profiling_raw.json"), "w") as f:
        #             json.dump(res_for_log.profiling_logs, f, indent=2)
        
        #         # 2. 解析并计算每层模块耗时
        #         from collections import defaultdict
        #         events_by_layer = defaultdict(dict)

        #         for event in res_for_log.profiling_logs:
        #             layer = event["layer_id"]
        #             etype = event["event"]
        #             ts = event["timestamp"]
        #             events_by_layer[layer][etype] = ts

        #         summary = {}
        #         summary_txt_lines = ["Layer | Attention Time (ms) | MLP Time (ms)", "-" * 45]

        #         for layer in sorted(events_by_layer.keys()):
        #             ev = events_by_layer[layer]
        #             attn_time = ev.get("attn_exit", 0) - ev.get("attn_enter", 0)
        #             mlp_time = ev.get("mlp_exit", 0) - ev.get("mlp_enter", 0)

        #             summary[f"layer_{layer}"] = {
        #                 "attn_time_sec": round(attn_time, 6),
        #                 "mlp_time_sec": round(mlp_time, 6)
        #             }

        #             summary_txt_lines.append(f"{layer:5} | {attn_time*1000:16.3f} | {mlp_time*1000:13.3f}")

        #         # 3. 保存结构化 JSON 摘要
        #         with open(os.path.join(log_dir, "summary.json"), "w") as f:
        #             json.dump(summary, f, indent=2)

        #         # 4. 保存人类可读的 TXT 摘要
        #         with open(os.path.join(log_dir, "summary.txt"), "w") as f:
        #             f.write("[Profiling Summary]\n")
        #             f.write("\n".join(summary_txt_lines))

        #     Logger(f"Full checkpoint saved to: {ckp_dir}")
        #     model.train()

def init_model(lm_config,loss_type="causal", resume_path=None):
    tokenizer = HFMathTokenizer()
    lm_config.vocab_size = tokenizer.vocab_size  # ← 新增这一行

    if loss_type == "causal":
        ModelClass = MiniMindForCausalLM
    elif loss_type == "mse":
        ModelClass = MiniMindForRegression
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    if resume_path is not None and os.path.exists(resume_path):
        Logger(f"Loading model from checkpoint: {resume_path}")
        model = ModelClass.from_pretrained(resume_path, config=lm_config)
    else:
        model = ModelClass(lm_config)

    model = model.to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer

def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """找最新且【完整】的 checkpoint 目录（跳过损坏/不完整 checkpoint）"""
    if not os.path.exists(save_dir):
        return None
    
    # 获取所有 checkpoint 目录（按时间倒序）
    ckpt_dirs = sorted(
        [d for d in glob.glob(os.path.join(save_dir, "checkpoint-*")) if os.path.isdir(d)],
        key=os.path.getctime,
        reverse=True
    )
    
    for ckpt_dir in ckpt_dirs:
        trainer_state_path = os.path.join(ckpt_dir, 'trainer_state.pth')
        # 检查文件是否存在 + 能否加载（轻量验证）
        if os.path.exists(trainer_state_path):
            try:
                # 尝试加载 state_dict meta（不加载权重，仅验证文件完整性）
                checkpoint = torch.load(trainer_state_path, map_location='cpu', weights_only=True)
                if 'model_state_dict' in checkpoint or 'optimizer_state_dict' in checkpoint:
                    print(f"✅ Found intact checkpoint: {ckpt_dir}")
                    return ckpt_dir
            except (EOFError, RuntimeError, zipfile.BadZipFile) as e:
                print(f"⚠️ Skipping corrupted checkpoint: {ckpt_dir} | Error: {e}")
                continue
    
    print("⚠️ No intact checkpoint found.")
    return None
    # if not os.path.exists(save_dir):
    #     return None
    # ckpt_dirs = [d for d in os.listdir(save_dir) if d.startswith("checkpoint-")]
    # if not ckpt_dirs:
    #     return None
    # ckpt_paths = [os.path.join(save_dir, d) for d in ckpt_dirs]
    # return max(ckpt_paths, key=os.path.getctime)


def resume_from_checkpoint(model, optimizer, scaler, checkpoint_path: str, device):
    """从 checkpoint 恢复模型、优化器、scaler 状态，返回 (start_epoch, start_step)"""
    ckpt_file = os.path.join(checkpoint_path, 'trainer_state.pth')
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"trainer_state.pth not found in {checkpoint_path}")
    
    Logger(f"Loading full checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(ckpt_file, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step']
    cumulative_train_time = checkpoint.get('cumulative_train_time', 0.0)  # New add
    Logger(f"Resumed training from epoch {start_epoch}, step {start_step}")
    return start_epoch, start_step, cumulative_train_time

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

def validate(model, val_loader, device, loss_type="causal"):
    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in val_loader:
            n_items = 0  # 🔥 关键：提前初始化 n_items，防 UnboundLocalError
            loss_val = 0.0

            try:
                if loss_type == "causal":
                    X, Y, loss_mask = batch
                    X, Y, loss_mask = X.to(device), Y.to(device), loss_mask.to(device)
                    res = model(input_ids=X)
                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(Y.size())
                    loss_val = (loss * loss_mask).sum().item()
                    n_items = loss_mask.sum().item()

                else:  # "mse"
                    input_ids, targets = batch
                    input_ids = input_ids.to(device)
                    targets = targets.to(device).squeeze(-1)
                    res = model(input_ids=input_ids, labels=targets)
                    # res.loss 是 batch-mean，还原为 sum
                    loss_val = (res.loss * targets.size(0)).item()
                    n_items = targets.size(0)

                total_loss += loss_val
                total_items += n_items

            except Exception as e:
                print(f"[Validate] Skip one batch due to error: {e}")
                continue  # 跳过坏 batch，不中断验证

    model.train()
    return total_loss / total_items if total_items > 0 else float('inf')
# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out")
    # 若要以最快速度实现zero则epochs设置为1轮；否则应当利用有限的数据训练2~6个epochs。
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=500,help="not use")
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="corpus/random/addition/1_digit_additions.txt")
    parser.add_argument("--resume_from", type=str, default=None,help="Path to a checkpoint directory to resume training from (e.g., ../out/checkpoint-epoch1-step1000)")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Validation split ratio (e.g., 0.05 for 5%)")
    parser.add_argument("--loss_type", type=str, default="causal", choices=["causal", "mse"],help="Loss type: 'causal' for language modeling (CE), 'mse' for regression")
    parser.add_argument("--save_every_n_epochs", type=int, default=500,
                    help="Save checkpoint every N epochs (default: 1). Last epoch always saved.")
    args = parser.parse_args()
    lm_config = MiniMindConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        max_position_embeddings=512,
        rope_theta=10000.0,
        use_moe=False,
        dropout=0,
        flash_attn=True
    )

    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # ========== DDP 和设备初始化 ==========
    ddp = int(os.environ.get("RANK", -1)) != -1  # 检测是否由 torchrun 启动

    if ddp:
        # 初始化分布式后端
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])

        # 每个进程绑定到对应的 GPU
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)

        # 同步随机种子（确保每个 rank 数据打乱不同但可复现）
        base_seed = 1337
        torch.manual_seed(base_seed + ddp_rank)
        torch.cuda.manual_seed(base_seed + ddp_rank)
    else:
        # 非 DDP 模式
        ddp_rank = 0
        ddp_local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)

    # 统一设置 args.device 供后续使用
    args.device = device


    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import swanlab as wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config, resume_path=args.resume_from, loss_type=args.loss_type)
    if not ddp or dist.get_rank() == 0:
        tokenizer_save_path = os.path.join(args.save_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_save_path)
        Logger(f"Tokenizer has been saved to: {tokenizer_save_path}")

    # ====== 🌟 新增：划分训练集和验证集 ======
    if args.loss_type == "causal":
        DatasetClass = PretrainDataset
    elif args.loss_type == "mse":
        DatasetClass = MSEPretrainDataset
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")

    full_dataset = DatasetClass(args.data_path, tokenizer, max_length=args.max_seq_len)
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_ratio)
    train_size = total_size - val_size

    # 固定随机种子确保可复现
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    Logger(f"Dataset split: {train_size} train, {val_size} val")

    # ====== 🌟 新增：保存数据集划分信息 ======
    if not ddp or (ddp and dist.get_rank() == 0):
        dataset_dir = os.path.join(args.save_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        # 保存划分索引（最轻量）
        split_info = {
            "train_indices": train_ds.indices,
            "val_indices": val_ds.indices,
            "val_ratio": args.val_ratio,
            "total_size": total_size,
            "data_path": args.data_path,
            "seed": 42
        }
        with open(os.path.join(dataset_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)

        # （可选）保存实际样本预览（用于人工检查）
        preview = {
            "train_samples": [tokenizer.decode(full_dataset[i][0], skip_special_tokens=False) for i in train_ds.indices[:5]],
            "val_samples": [tokenizer.decode(full_dataset[i][0], skip_special_tokens=False) for i in val_ds.indices[:5]]
        }
        with open(os.path.join(dataset_dir, "samples_preview.json"), "w") as f:
            json.dump(preview, f, indent=2, ensure_ascii=False)

        Logger(f"Dataset split info saved to: {dataset_dir}")

    if args.loss_type == "causal":
        DatasetClass = PretrainDataset
    elif args.loss_type == "mse":
        DatasetClass = MSEPretrainDataset
    else:
        raise ValueError(f"Unknown loss_type: {args.loss_type}")
    full_dataset = DatasetClass(args.data_path, tokenizer, max_length=args.max_seq_len)
    Logger(f"✅ Loaded {args.loss_type} dataset with {len(full_dataset)} samples")
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        sampler=train_sampler
    )
    # ====== 🌟 新增：验证集 DataLoader ======
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # ========== 在 DDP 之前恢复 checkpoint ==========
    actual_model = model  # 保存原始模型引用
    start_epoch, start_step_in_epoch = 0, 0
    cumulative_train_time = 0.0  # ✅ 初始化累计时间

    resume_checkpoint = args.resume_from or find_latest_checkpoint(args.save_dir)
    if resume_checkpoint:
        try:
            start_epoch, start_step_in_epoch, cumulative_train_time = resume_from_checkpoint(
                actual_model, optimizer, scaler, resume_checkpoint, args.device
            )
        except Exception as e:
            Logger(f"Resume failed: {e}. Starting from scratch.")
            cumulative_train_time = 0.0  # 出错则归零
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    # ====== 🌟 新增：记录训练开始时间 ======
    this_run_start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        current_start_step = start_step_in_epoch if epoch == start_epoch else 0
        current_start_step = start_step_in_epoch if epoch == start_epoch else 0
        train_epoch(
            epoch, 
            wandb, 
            current_start_step, 
            iter_per_epoch, 
            this_run_start_time, 
            cumulative_train_time
        )
        # ====== 🌟 新增：每 epoch 验证 ======
        if not ddp or dist.get_rank() == 0:
            val_loss = validate(model, val_loader, args.device, args.loss_type)
            Logger(f"Epoch {epoch+1} Validation Loss: {val_loss:.6f}")
            if wandb is not None:
                wandb.log({"val_loss": val_loss, "epoch": epoch+1})  

            # === 📝 新增：本地持久化 val loss 日志（CSV格式）===
            val_loss_log_path = os.path.join(args.save_dir, "val_loss.csv")
            header_needed = not os.path.exists(val_loss_log_path)
            with open(val_loss_log_path, "a") as f_val:
                if header_needed:
                    f_val.write("epoch,val_loss\n")
                f_val.write(f"{epoch + 1},{val_loss:.8f}\n")
            # === 📝 新增结束 ===
        # =============== 📦 Epoch-Level Checkpoint Save ===============
        should_save = (
            (epoch + 1) % args.save_every_n_epochs == 0   # 满足周期
            or (epoch + 1) == args.epochs                  # 或是最后一轮
        )
        if should_save and (not ddp or dist.get_rank() == 0):
            model.eval()
            Logger(f"📦 Saving checkpoint for epoch {epoch + 1}...")

            # 🔥 用验证集第一个 batch 做 profiling（稳定+非训练污染）
            try:
                val_iter = iter(val_loader)
                batch_for_prof = next(val_iter)
                if args.loss_type == "causal":
                    X, _, _ = batch_for_prof
                    profiling_input = X[:1].to(args.device)
                else:  # mse
                    input_ids, _ = batch_for_prof
                    profiling_input = input_ids[:1].to(args.device)
            except:
                # fallback: dummy input
                profiling_input = torch.randint(0, lm_config.vocab_size, (1, args.max_seq_len), device=args.device)

            # 做一次带 profiling 的前向
            with torch.no_grad(), ctx:
                res_for_log = model(profiling_input, profiling=True)

            # 构建路径
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            ckp_dir = os.path.join(args.save_dir, f"checkpoint-epoch{epoch+1}-{timestamp}")
            os.makedirs(ckp_dir, exist_ok=True)

            # 保存模型（HF 格式）
            actual_model = model.module if isinstance(model, DistributedDataParallel) else model
            actual_model.save_pretrained(ckp_dir, safe_serialization=False)

            # 更新时间
            current_this_run_elapsed = time.time() - this_run_start_time
            current_total_time = cumulative_train_time + current_this_run_elapsed

            # 保存 trainer_state
            checkpoint = {
                'epoch': epoch + 1,
                'step': 0,  # 下一 epoch 从 step 0 开始
                'model_state_dict': actual_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'lr': optimizer.param_groups[0]['lr'],
                'args_dict': vars(args),
                'lm_config_dict': lm_config.to_dict() if hasattr(lm_config, 'to_dict') else lm_config.__dict__,
                'cumulative_train_time': current_total_time,
            }
            torch.save(checkpoint, os.path.join(ckp_dir, 'trainer_state.pth'))

            # === 保存 profiling 日志（复用你原有逻辑，无改动）===
            if hasattr(res_for_log, 'profiling_logs') and res_for_log.profiling_logs:
                log_dir = os.path.join(ckp_dir, "log")
                os.makedirs(log_dir, exist_ok=True)
                from collections import defaultdict
                events_by_layer = defaultdict(dict)
                for event in res_for_log.profiling_logs:
                    layer = event["layer_id"]
                    etype = event["event"]
                    ts = event["timestamp"]
                    events_by_layer[layer][etype] = ts
                summary = {}
                summary_txt_lines = ["Layer | Attention Time (ms) | MLP Time (ms)", "-" * 45]
                for layer in sorted(events_by_layer.keys()):
                    ev = events_by_layer[layer]
                    attn_time = ev.get("attn_exit", 0) - ev.get("attn_enter", 0)
                    mlp_time = ev.get("mlp_exit", 0) - ev.get("mlp_enter", 0)
                    summary[f"layer_{layer}"] = {
                        "attn_time_sec": round(attn_time, 6),
                        "mlp_time_sec": round(mlp_time, 6)
                    }
                    summary_txt_lines.append(f"{layer:5} | {attn_time*1000:16.3f} | {mlp_time*1000:13.3f}")
                with open(os.path.join(log_dir, "profiling_raw.json"), "w") as f:
                    json.dump(res_for_log.profiling_logs, f, indent=2)
                with open(os.path.join(log_dir, "summary.json"), "w") as f:
                    json.dump(summary, f, indent=2)
                with open(os.path.join(log_dir, "summary.txt"), "w") as f:
                    f.write("[Profiling Summary]\n")
                    f.write("\n".join(summary_txt_lines))

            Logger(f"✅ Checkpoint saved to: {ckp_dir}")
            model.train()
    # ====== 🌟 新增：记录训练结束时间并打印总耗时 ======
    this_run_end_time = time.time()
    this_run_duration = this_run_end_time - this_run_start_time
    total_train_time = cumulative_train_time + this_run_duration
    # 格式化为易读形式
    hours, rem = divmod(total_train_time, 3600)
    minutes, seconds = divmod(rem, 60)
    Logger(f"✅ Total training time: {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s")
    Logger(f"✅ Total training time (seconds): {total_train_time:.2f}")
    if not ddp or (ddp and dist.get_rank() == 0):
        time_log = {
            "total_seconds": total_train_time,
            "formatted": f"{int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s",
            "epochs_trained": args.epochs - start_epoch,
            "total_steps": (args.epochs - start_epoch) * iter_per_epoch,
        }
        time_log_path = os.path.join(args.save_dir, "training_time.json")
        with open(time_log_path, "w") as f:
            json.dump(time_log, f, indent=2)
        Logger(f"✅ Training time logged to: {time_log_path}")
    
    if wandb is not None and (not ddp or dist.get_rank() == 0):
        wandb.log({
            "total_train_time_sec": total_train_time,
            "total_train_time_formatted": f"{int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s"
        })
    # 训练循环结束后
    if not ddp or (ddp and dist.get_rank() == 0):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_name = f"minimind-math-h{args.hidden_size}-l{args.num_hidden_layers}"
        if args.use_moe:
            model_name += "-moe"
        model_name += f"-{timestamp}"

        final_model_path = os.path.join(args.save_dir, model_name)
        os.makedirs(final_model_path, exist_ok=True)

        actual_model = model.module if isinstance(model, DistributedDataParallel) else model
        actual_model.save_pretrained(final_model_path, safe_serialization=False)
        tokenizer.save_pretrained(final_model_path)

        Logger(f"The final model has been saved to: {final_model_path}")