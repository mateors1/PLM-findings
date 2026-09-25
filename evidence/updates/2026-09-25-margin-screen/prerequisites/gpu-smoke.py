"""Tiny synthetic CUDA/BF16 numerical smoke; no research data or quality gate."""
import hashlib
import json
from pathlib import Path

import torch

from plm.config import ModelConfig
from plm.model import build_model

root=Path.cwd(); out=Path(__file__).with_name('gpu-smoke.json')
assert not out.exists() and torch.cuda.is_available()
names=['src/plm/config.py','src/plm/model/layers.py','src/plm/model/interfaces.py',
       'src/plm/training/trainer.py','src/plm/configuration.py','src/plm/serving/runtime.py']
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
inputs={name:sha(root/name) for name in names}
torch.manual_seed(9091); torch.cuda.manual_seed_all(9091)
config=ModelConfig(vocab_size=1030,max_seq_len=16,dim=32,n_layers=2,n_heads=4,
    n_kv_heads=2,ffn_multiple_of=8,dtype='bf16',prompt_set_loss_weight=1.0,
    symmetric_relation_loss_weight=1.0,symmetric_margin_loss_weight=0.1,symmetric_margin=1.0)
model=build_model(config).to('cuda').train()
tokens=torch.tensor([[1,1024,32,34,5,1025,1026,2],[1,1026,33,34,5,1027,2,0]],device='cuda')
labels=tokens.clone(); labels[:,:5]=-100; mask=tokens!=0
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,fused=True)
rows=[]
for step in range(3):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result=model(tokens,labels=labels,attention_mask=mask)
    assert result.symmetric_margin_loss.dtype==torch.float32
    assert result.symmetric_relation_logits.dtype==torch.float32
    assert result.symmetric_margin_query_count==2
    assert torch.isfinite(result.loss) and torch.isfinite(result.symmetric_margin_loss)
    result.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step()
    rows.append({'step':step+1,'loss':result.loss.item(),'margin':result.symmetric_margin_loss.item(),
                 'active_queries':result.symmetric_margin_active_count})
torch.cuda.synchronize()
assert inputs=={name:sha(root/name) for name in names}
receipt={'complete':True,'scope':'Three synthetic CUDA BF16 optimization steps, FP32 margin and finite gradients; no dataset or quality measurement.',
         'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
         'model_config':config.model_dump(mode='json'),'steps':rows,'input_sha256':inputs,
         'script_sha256':sha(__file__)}
with out.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'complete':True,'receipt_sha256':sha(out),'steps':rows}))
