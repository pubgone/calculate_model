import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 在 import torch 之前设置！
import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from typing import Optional
from my_tokenizers.hf_math_tokenizer import HFMathTokenizer
from models.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model_ultils.pretrain_dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb,start_step, iter_per_epoch):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        if step < start_step:
            continue  # 跳过已训练的 steps
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # ✅ 使用传入的 iter_per_epoch 计算全局 step
        global_step = epoch * iter_per_epoch + step
        total_steps = args.epochs * iter_per_epoch
        lr = get_lr(global_step, total_steps, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
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
        if ((step + 1) % args.save_interval == 0 or step == iter_per_epoch - 1) and (not ddp or dist.get_rank() == 0):
            model.eval()
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            ckp_dir = os.path.join(args.save_dir, f"checkpoint-epoch{epoch+1}-step{step+1}-{timestamp}")
            os.makedirs(ckp_dir, exist_ok=True)

            actual_model = model.module if isinstance(model, DistributedDataParallel) else model
            actual_model.save_pretrained(ckp_dir, safe_serialization=False)
            # === 新增：保存训练状态 ===
            checkpoint = {
                'epoch': epoch,
                'step': step + 1,
                'model_state_dict': actual_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
                'lr': optimizer.param_groups[0]['lr'],
                'args_dict': vars(args),            # 安全！
                'lm_config_dict': lm_config.to_dict() if hasattr(lm_config, 'to_dict') else lm_config.__dict__,
            }
            torch.save(checkpoint, os.path.join(ckp_dir, 'trainer_state.pth'))

            Logger(f"Full checkpoint saved to: {ckp_dir}")
            model.train()
        # if ((step + 1) % args.save_interval == 0 or step == iter_per_epoch - 1) and (not ddp or dist.get_rank() == 0):
        #     model.eval()
        #     ckp_dir = os.path.join(args.save_dir, f"checkpoint-epoch{epoch+1}-step{step+1}")
        #     os.makedirs(ckp_dir, exist_ok=True)

        #     actual_model = model.module if isinstance(model, DistributedDataParallel) else model

        #     # 只保存模型（config + weights）
        #     actual_model.save_pretrained(ckp_dir, safe_serialization=False)
        #     # 不再保存 tokenizer！

        #     Logger(f"Model checkpoint has been saved to: {ckp_dir}")
        #     model.train()    

        # if ((step + 1) % args.save_interval == 0 or step == iter_per_epoch - 1) and (not ddp or dist.get_rank() == 0):
        #     model.eval()
        #     moe_path = '_moe' if lm_config.use_moe else ''
        #     ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

        #     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        #         state_dict = model.module.state_dict()
        #     else:
        #         state_dict = model.state_dict()

        #     state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
        #     torch.save(state_dict, ckp)
        #     model.train()


def init_model(lm_config, resume_path=None):
    tokenizer = HFMathTokenizer()
    lm_config.vocab_size = tokenizer.vocab_size  # ← 新增这一行
    if resume_path is not None and os.path.exists(resume_path):
        Logger(f"Loading model from checkpoint: {resume_path}")
        model = MiniMindForCausalLM.from_pretrained(resume_path, config=lm_config)
    else:
        model = MiniMindForCausalLM(lm_config)

    model = model.to(args.device)
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer

def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    if not os.path.exists(save_dir):
        return None
    ckpt_dirs = [d for d in os.listdir(save_dir) if d.startswith("checkpoint-")]
    if not ckpt_dirs:
        return None
    ckpt_paths = [os.path.join(save_dir, d) for d in ckpt_dirs]
    return max(ckpt_paths, key=os.path.getctime)


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
    Logger(f"Resumed training from epoch {start_epoch}, step {start_step}")
    return start_epoch, start_step

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


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
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="corpus/random/addition/1_digit_additions.txt")
    parser.add_argument("--resume_from", type=str, default=None,
                    help="Path to a checkpoint directory to resume training from (e.g., ../out/checkpoint-epoch1-step1000)")
    args = parser.parse_args()
    lm_config = MiniMindConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        max_position_embeddings=512,
        rope_theta=10000.0,
        use_moe=False,
        dropout=0.1,
        flash_attn=True
    )
    # lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
    #                            use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    # model, tokenizer = init_model(lm_config)
    # print("✅ Tokenizer vocab size:", tokenizer.vocab_size)
    # print("✅ Model vocab size:", model.config.vocab_size)
    # print("✅ Test encode:", tokenizer("12+34=46", add_special_tokens=True).input_ids)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import swanlab as wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config, resume_path=args.resume_from)
    if not ddp or dist.get_rank() == 0:
        tokenizer_save_path = os.path.join(args.save_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_save_path)
    Logger(f"Tokenizer has been saved to: {tokenizer_save_path}")
    # # 🔍 === 验证代码放在这里 ===
    # if not ddp or dist.get_rank() == 0:  # 只在主进程打印，避免多卡重复输出
    #     print("✅ Tokenizer vocab size:", tokenizer.vocab_size)
    #     print("✅ Model vocab size:", model.config.vocab_size)
    #     test_output = tokenizer("12+34=46", add_special_tokens=True, return_tensors="pt")
    #     test_encode = tokenizer("12+34=46", add_special_tokens=True).input_ids
    #     print("✅ Test encode shape:", test_encode.shape)        # torch.Size([1, L])
    #     print("✅ Test encode IDs:", test_encode.tolist())       # 转为 list 打印
    #     print("✅ Decoded back:", tokenizer.decode(test_encode[0], skip_special_tokens=False))
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
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

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # ========== 在 DDP 之前恢复 checkpoint ==========
    actual_model = model  # 保存原始模型引用
    start_epoch, start_step_in_epoch = 0, 0
    resume_checkpoint = args.resume_from or find_latest_checkpoint(args.save_dir)
    if resume_checkpoint:
        try:
            start_epoch, start_step_in_epoch = resume_from_checkpoint(
                actual_model, optimizer, scaler, resume_checkpoint, args.device
            )
        except Exception as e:
            Logger(f"Resume failed: {e}. Starting from scratch.")
    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        current_start_step = start_step_in_epoch if epoch == start_epoch else 0
        train_epoch(epoch, wandb, start_step_in_epoch if epoch == start_epoch else 0, iter_per_epoch)
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
    # if not ddp or (ddp and dist.get_rank() == 0):
    #     # 构建最终模型名称
    #     model_name = f"minimind-math-h{args.hidden_size}-l{args.num_hidden_layers}"
    #     if args.use_moe:
    #         model_name += "-moe"

    #     final_model_path = os.path.join(args.save_dir, model_name)
    #     os.makedirs(final_model_path, exist_ok=True)

    #     actual_model = model.module if isinstance(model, DistributedDataParallel) else model

    #     # 保存完整模型（权重 + config）
    #     actual_model.save_pretrained(final_model_path, safe_serialization=False)
    #     # 保存 tokenizer（这次要包含！）
    #     tokenizer.save_pretrained(final_model_path)

    #     Logger(f"The final model has been saved to: {final_model_path}")
