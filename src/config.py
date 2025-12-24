from typing import Any, Dict, List, Union

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str
    n_tasks: int
    eval_ratio: float
    test_ratio: float
    seed: int


class ModelConfig(BaseModel):
    type: str
    input_dim: int
    hidden_dims: Union[int, List[int]]
    optimizer: str
    lr: float
    criterion: str


class HyperparamStrategyConfig(BaseModel):
    name: str
    n_calls: int
    params: Dict[str, Any]


class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    strategies: List[HyperparamStrategyConfig]
