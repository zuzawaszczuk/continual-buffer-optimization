import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SimpleMLP(num_classes=10)

perm_mnist = PermutedMNIST(n_experiences=5)
train_stream = perm_mnist.train_stream
test_stream = perm_mnist.test_stream

optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = CrossEntropyLoss()

interactive_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(stream=True),
    loss_metrics(stream=True),
    loggers=[InteractiveLogger()]
)

cl_strategy = Naive(
    model=model, 
    optimizer=optimizer, 
    criterion=criterion, 
    train_mb_size=32, 
    train_epochs=2, 
    eval_mb_size=32, 
    evaluator=eval_plugin, 
    device=device
)

results = []
for train_task in train_stream:
    cl_strategy.train(train_task, num_workers=4)
    results.append(cl_strategy.eval(test_stream))

for key in results:
    print(key)