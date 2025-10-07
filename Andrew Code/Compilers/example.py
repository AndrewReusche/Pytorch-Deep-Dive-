import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

model = Model()
x = torch.randn(10)

torch._dynamo.config.compiled_autograd = True

@torch.compile
def train(model, x):
    model = torch.compile(model)
    loss = model(x).sum()
    torch._dynamo.config.compiled_autograd = True
    torch.compile(lambda: loss.backward(), fullgraph=True)()

train(model, x)

