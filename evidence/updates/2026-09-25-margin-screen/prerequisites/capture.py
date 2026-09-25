"""Compare behavior through this process's explicitly selected source tree."""
import argparse
import hashlib
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--source-root', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source_root.resolve() / 'src'))
import torch
from plm.config import ModelConfig
from plm.model import build_model
import plm.model.layers

assert not args.out.exists()
assert Path(plm.model.layers.__file__).resolve().is_relative_to(args.source_root.resolve())
torch.set_num_threads(1)
torch.manual_seed(9091)
config = ModelConfig(vocab_size=1030, max_seq_len=16, dim=32, n_layers=2,
    n_heads=4, n_kv_heads=2, ffn_multiple_of=8, dtype='fp32',
    resid_dropout=0.15, embed_dropout=0.15,
    prompt_set_loss_weight=1.0, symmetric_relation_loss_weight=1.0)
model = build_model(config).train()
result = {'initial_state': {k:v.detach().clone() for k,v in model.state_dict().items()},
          'initial_rng':torch.get_rng_state().clone()}
tokens = torch.tensor([[1,1024,32,34,5,1025,1026,1025,2,0],
                       [1,1026,33,34,5,1027,2,0,0,0]])
mask = tokens != 0
labels = tokens.clone(); labels[:,:5] = -100
out = model(tokens,labels=labels,attention_mask=mask)
assert getattr(out,'symmetric_margin_loss',None) is None
assert getattr(out,'symmetric_margin_query_count',0)==0
for field in ('logits','loss','task_loss','first_target_loss','prompt_set_logits',
              'prompt_set_loss','symmetric_relation_logits','symmetric_relation_loss'):
    result[field]=getattr(out,field).detach().clone()
out.loss.backward()
result['gradients']={k:None if v.grad is None else v.grad.detach().clone() for k,v in model.named_parameters()}
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
optimizer.step()
result['updated_state']={k:v.detach().clone() for k,v in model.state_dict().items()}
result['final_rng']=torch.get_rng_state().clone()
model.eval()
with torch.inference_mode():
    result['unlabeled_logits']=model(tokens[:,:5]).logits.clone()
    result['cached_logits']=model.decode(tokens[:,:5]).logits.clone()
torch.save(result,args.out)
print(args.out.name,hashlib.sha256(args.out.read_bytes()).hexdigest())
