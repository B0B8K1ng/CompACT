"""Check the actual rollout loops feed encoded RGB, never generated latent."""
import ast
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
def method(path,cls,name,namespace):
 tree=ast.parse((ROOT/path).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
 fn=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
 return namespace[name]

class Config(dict):
 __getattr__=dict.__getitem__

def test_nwm_pixel_feedback_encodes_rgb_window():
 seen=[]
 def forward(models,context,*args,**kwargs):
  seen.append(context.clone());return torch.full((1,1,1,1),3.)
 vae=SimpleNamespace(encode=lambda x:x*2,decode=lambda x,denormalize=True:x/10,get_code_from_latent=lambda x:x)
 obj=SimpleNamespace(config=Config(planning_image_feedback=True,planning_microbatch_size=1,rollout_stride=1,seed=42,dataset=SimpleNamespace(mean=[.5],std=[.5])),num_samples=1,num_cond=4,device='cpu',latent_size=1,vae=vae,model=None,diffusion=None,_real_motion_for_model=lambda x:x)
 fn=method('planning_eval.py','WM_Planning_Evaluator','autoregressive_rollout_without_intermediate_decoding',dict(torch=torch,tqdm=lambda x:x,model_forward_wrapper=forward))
 result,_=fn(obj,torch.zeros(1,4,1,1,1),torch.zeros(1,2,3),1,only_final=True)
 torch.testing.assert_close(seen[1][:,-1],torch.full((1,1,1,1),-.8))
 torch.testing.assert_close(result,torch.full((1,1,1,1,1),.3))

def test_rae_pixel_feedback_keeps_raw_final_prediction_for_cost():
 seen=[]
 def sample(noise,model,**kwargs):
  seen.append(kwargs['x_cond'].clone());return [torch.full((1,1,1,1),3.)]
 obj=SimpleNamespace(args=SimpleNamespace(image_feedback=True,microbatch_size=1,seed=42),device='cpu',rae=SimpleNamespace(latent_dim=1),latent_size=1,model=None,sample_fn=sample,decode=lambda x,size:x/10,encode=lambda x:x*2)
 fn=method('scripts/raenwm_planning_eval.py','Planner','rollout_final',dict(torch=torch,CONTEXT_SIZE=4,samplewise_randn=lambda keys,shape,**kw:torch.zeros(len(keys),*shape)))
 with patch('torch.amp.autocast',side_effect=lambda *args,**kw:nullcontext()):
  final=fn(obj,torch.zeros(1,4,1,1,1),torch.zeros(1,2,3),['case'],'test')
 torch.testing.assert_close(seen[1][:,-1],torch.full((1,1,1,1),-.8))
 torch.testing.assert_close(final,torch.full((1,1,1,1),3.))
