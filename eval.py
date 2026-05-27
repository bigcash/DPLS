import os
import time
import math
import pickle
from contextlib import nullcontext
import importlib
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.nn.functional as F
from datetime import datetime
import json

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on Fineweb-edu 100B
# I/O
data_path = "data"
out_dir = 'output/out'
resume_dir = '.'
change_criterion = True
eval_interval = 2000
log_interval = 1
eval_iters = 1000
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
# init_from = 'resume'
# swanlab logging
swanlab_log = False  # disabled by default
swanlab_project = 'T6'
swanlab_run_name = 'gpt2'  # 'run' + str(time.time())
# data
dataset = 'fineweb-edu100B'
gradient_accumulation_steps = 5  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
head_dim = 64
rank = 2
q_rank = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
using_groupnorm = False
dpls_epsilon = 0.2
dpls_top_k = 100
dpls_ignore_index = -1
# optimizer
optimizer_name = 'adamw'
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
rho = 0.1
interval = 10
variant = 4
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
schedule = 'cosine'
model_type = 'base_model'
group_size = 11
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
scale_attn_by_inverse_layer_idx = True
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------
model_file = importlib.import_module(f'model.{model_type}')
GPTConfig = model_file.GPTConfig
GPT = model_file.GPT


def get_num_params(self, non_embedding=False):
    """
    Return the number of parameters in the model.
    For non-embedding count (default), the position embeddings get subtracted.
    The token embeddings would too, except due to the parameter sharing these
    params are actually used as weights in the final layer, so we include them.
    """
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding:
        n_params -= self.transformer.wpe.weight.numel()
    return n_params


# Get current date and job ID
current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id = os.environ.get('SLURM_JOB_ID', '0')

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    print(
        f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}, RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    world_size = 1
    gradient_accumulation_steps *= 8  # simulate 8 gpus

# Calculate total tokens in billions
tokens_per_iter = batch_size * block_size * gradient_accumulation_steps * world_size
total_tokens_B = tokens_per_iter * max_iters / (1000 ** 3)

# Add after the initial variable declarations
tokens_trained = 0  # track total tokens trained

# Initialize random seed and torch settings
torch.manual_seed(5000 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

# Poor man's data loader
data_dir = os.path.join(data_path, dataset)
# train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
train_file_list = list(filter(lambda x: x.endswith('.bin') and x.startswith('fineweb_train'), os.listdir(data_dir)))
train_data_list = [np.memmap(os.path.join(data_dir, file), dtype=np.uint16, mode='r') for file in train_file_list]
val_data = np.memmap(os.path.join(data_dir, 'fineweb_val_000000.bin'), dtype=np.uint16, mode='r')
import random

random.seed(5000 + seed_offset)


def get_batch(split):
    if split == 'train':
        data = random.choice(train_data_list)
    else:
        data = val_data
    offset = 512
    ix = torch.randint(len(data) - block_size - offset, (batch_size,))
    x = torch.stack([torch.from_numpy((data[offset + i:offset + i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[offset + i + 1:offset + i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# Init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# Attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# Model initialization
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, head_dim=head_dim, rank=rank, q_rank=q_rank, using_groupnorm=using_groupnorm,
                  vocab_size=None, dropout=dropout, scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
                  dpls_epsilon=dpls_epsilon, dpls_top_k=dpls_top_k,
                  dpls_ignore_index=dpls_ignore_index)  # start with model_args from command line
if "gqa" in model_type:
    model_args['group_size'] = group_size
if init_from == 'resume':
    print(f"Resuming training from {resume_dir}")
    # Resume training from a checkpoint.
    # ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    # checkpoint = torch.load(ckpt_path, map_location=device)
    # checkpoint_model_args = checkpoint['model_args']
    config = GPTConfig.from_json_file(os.path.join(resume_dir, 'config.json'))
    model = GPT.from_pretrained(resume_dir, config=config)

    # Force these config attributes to be equal otherwise we can't even resume training
    # The rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(config, k)
    model.transformer.wte.weight = model.lm_head.weight
else:
    print("init model error")
    exit(0)
# Crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value
model.to(device)

# Now calculate non-embedding parameters
param_count = get_num_params(model, non_embedding=False)
param_count_m = param_count / 1_000_000  # convert to millions

params = list(model.parameters())
# Compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# Helps estimate an arbitrarily accurate loss over either split using many batches
raw_model = model.module if ddp else model # unwrap DDP container if needed
print(f"eval_iters*batch_size*block_size={eval_iters}*{batch_size}*{block_size}={eval_iters*batch_size*block_size}")

def compute_avg_entropy(log_p, p):
    entropy = -torch.sum(p * log_p, dim=-1)
    return entropy.mean()


def compute_topk_entropy(log_p, p, top_k=10, eps=1e-9):
    topk_probs, _ = torch.topk(p, top_k, dim=-1)  # [B, S-1, top_k]
    normalizer = topk_probs.sum(dim=-1, keepdim=True)  # [B, S-1, 1]
    topk_probs_norm = topk_probs / (normalizer + eps)
    log_topk = torch.log(topk_probs_norm + eps)
    entropy_topk = -torch.sum(topk_probs_norm * log_topk, dim=-1)  # [B, S-1]
    return entropy_topk.mean()

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        avg_entropies = np.zeros(eval_iters)
        top10_entropies = np.zeros(eval_iters)
        top100_entropies = np.zeros(eval_iters)
        top1000_entropies = np.zeros(eval_iters)
        top10000_entropies = np.zeros(eval_iters)
        for k in tqdm(range(eval_iters)):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            next_token_logits = logits[0, :-1, :]  # [seq_len-1, vocab]
            log_p = F.log_softmax(next_token_logits, dim=-1)
            p = torch.exp(log_p)
            avg_entropy = compute_avg_entropy(log_p, p).cpu().numpy()
            top10_entropy = compute_topk_entropy(log_p, p, top_k=10).cpu().numpy()
            top100_entropy = compute_topk_entropy(log_p, p, top_k=100).cpu().numpy()
            top1000_entropy = compute_topk_entropy(log_p, p, top_k=1000).cpu().numpy()
            top10000_entropy = compute_topk_entropy(log_p, p, top_k=10000).cpu().numpy()
            avg_entropies[k] = avg_entropy
            top10_entropies[k] = top10_entropy
            top100_entropies[k] = top100_entropy
            top1000_entropies[k] = top1000_entropy
            top10000_entropies[k] = top10000_entropy
        loss_mean = losses.mean()
        perplexity = math.exp(loss_mean)
        out[split] = {
            'loss': loss_mean,
            'perplexity': perplexity,
            'avg_entropy': avg_entropies.mean(),
            'top10_entropy': top10_entropies.mean(),
            'top100_entropy': top100_entropies.mean(),
            'top1000_entropy': top1000_entropies.mean(),
            'top10000_entropy': top10000_entropies.mean()
        }
    model.train()
    return out

losses = estimate_loss()
print(f"train loss {losses['train']['loss']:.4f}, perplexity {losses['train']['perplexity']:.4f}, avg entro {losses['train']['avg_entropy']:.4f}, top10 entro {losses['train']['top10_entropy']:.4f}, top100 entro {losses['train']['top100_entropy']:.4f}, top1000 entro {losses['train']['top1000_entropy']:.4f}, top10000 entro {losses['train']['top10000_entropy']:.4f}")
print(f"val loss {losses['val']['loss']:.4f}, perplexity {losses['val']['perplexity']:.4f}, avg entro {losses['val']['avg_entropy']:.4f}, top10 entro {losses['val']['top10_entropy']:.4f}, top100 entro {losses['val']['top100_entropy']:.4f}, top1000 entro {losses['val']['top1000_entropy']:.4f}, top10000 entro {losses['val']['top10000_entropy']:.4f}")
