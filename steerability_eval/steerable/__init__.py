from typing import Type
from steerability_eval.steerable.base import BaseSteerableSystem
from steerability_eval.steerable.few_shot import FewShotSteerable
from steerability_eval.steerable.honcho import AsyncHonchoSteerable, HonchoSteerable

def get_steerable_system_class(name: str) -> Type[BaseSteerableSystem]:
    if name == 'FewShotSteerable':
        return FewShotSteerable
    elif name == 'AsyncHonchoSteerable':
        return AsyncHonchoSteerable
    elif name == 'HonchoSteerable':
        return HonchoSteerable
    else:
        raise ValueError(f'Unknown steerable system: {name}')
