"""Unavailable windows must not renumber qualitative navigation samples."""
import ast
import pickle
import tempfile
from pathlib import Path


class Config(dict):
    __getattr__ = dict.__getitem__


def test_navigation_source_ids_survive_filtering():
    tree=ast.parse((Path(__file__).resolve().parents[1]/'planning_eval.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='get_dataset_eval')
    original=[('missing',3,8,8),('valid_a',7,8,8),('valid_b',9,8,8)]
    class Dataset:
        def __init__(self,**kwargs):self.index_to_data=original[1:]
        def __len__(self):return len(self.index_to_data)
    with tempfile.TemporaryDirectory() as directory:
        split=Path(directory)/'index.pkl';split.write_bytes(pickle.dumps(original))
        data=Config(navigation_index=str(split),data_folder='',test='',metric_waypoint_spacing=.25,navigation_sample_count=3)
        cfg=Config(planning_preserve_source_indices=True,planning_sample_indices=[2],dataset=Config(image_size=224,normalize=True,mean=[.5]*3,std=[.5]*3,action_stats={}),trajectory_eval_distance=Config(min_dist_cat=8,max_dist_cat=8),trajectory_eval_len_traj_pred=8,traj_stride=1,trajectory_eval_context_size=4)
        scope=dict(pickle=pickle,TrajectoryEvalDataset=Dataset,get_transform=lambda *args:None,get_planning_data_config=lambda *args:data)
        exec(compile(ast.Module(body=[function],type_ignores=[]),'planning_eval.py','exec'),scope)
        get_dataset=scope['get_dataset_eval']
        result=get_dataset(cfg,'huron')
        assert result.index_to_data[2]==original[2] and len(result)==3
        for requested in [[0],[3],[-1],None]:
            cfg['planning_sample_indices']=requested
            try:get_dataset(cfg,'huron')
            except ValueError:pass
            else:raise AssertionError(f'Invalid selection accepted: {requested}')
        cfg['planning_preserve_source_indices']=False
        try:get_dataset(cfg,'huron')
        except ValueError:pass
        else:raise AssertionError('Default strict count validation was bypassed')
