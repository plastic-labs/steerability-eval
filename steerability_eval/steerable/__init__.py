from typing import Type
from steerability_eval.steerable.base import BaseSteerableSystem
from steerability_eval.steerable.few_shot import FewShotSteerableNoPersona

def get_steerable_system_class(name: str) -> Type[BaseSteerableSystem]:
    if name == 'FewShotSteerableNoPersona':
        return FewShotSteerableNoPersona
    else:
        raise ValueError(f'Unknown steerable system: {name}')