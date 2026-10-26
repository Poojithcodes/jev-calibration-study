from types import SimpleNamespace
from jevcal.setfit_sampling import bounded_trainer_class


def test_pair_limit_keeps_full_emitted_prefix_and_existing_tighter_limit():
    class Base:
        def get_dataset(self,x,y,args,max_pairs=-1):
            return max_pairs
    trainer=bounded_trainer_class(Base)()
    args=SimpleNamespace(num_iterations=20)
    assert trainer.get_dataset(range(9600),[],args)==384000
    assert trainer.get_dataset(range(9600),[],args,max_pairs=100)==100
    args.num_iterations=None
    assert trainer.get_dataset(range(9600),[],args)==-1
