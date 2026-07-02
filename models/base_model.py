from abc import ABC, abstractmethod
from typing import List, Dict, Any
from argparse import Namespace, ArgumentParser


class BaseModelWrapper(ABC):

    @staticmethod
    def get_args_dict() -> Dict[str, List[Any]]:
        '''Return a dictionary of argument names and their default values.'''
        raise NotImplementedError("Subclasses must implement get_args_dict property. if even it's empty, return {}")    
    
    @classmethod
    def get_relevant_args(self, args : Namespace, parser : ArgumentParser, notebook=False) -> Namespace:
            for key, value in self.get_args_dict().items():
                if not hasattr(args, key):
                    if type(value) == bool:
                        parser.add_argument(f'--{key}', action='store_true' if not value else 'store_false')
                    elif type(value) == list:
                        parser.add_argument(f'--{key}', type=type(value[0]), default=value, nargs='+')
                    else:
                        parser.add_argument(f'--{key}', type=type(value), default=value)
                else:
                    raise ValueError(f"Argument {key} already exists in args.")
            if notebook:
                new_args = parser.parse_args([])   
                # copy values from the provided args into the new namespace
                for key, value in vars(args).items():
                    if hasattr(new_args, key):
                        new_args.__setattr__(key, value)

            else:  
                new_args = parser.parse_args()
            return new_args
    

    def setup(self, args=None):
        pass


    
    @abstractmethod
    def load_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, batch: List[Dict[str, any]], max_new_tokens: int = 50, **generate_kwargs) -> List[str]:
        pass


