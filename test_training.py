import torch
from networks import ResNet18, MultiHeadedNetwork


net = MultiHeadedNetwork(
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    common = ResNet18(in_channels=1),
    head_codes = [[(131, 128), "R", (128, 1)]],
    lr=1e-4,
)

N = 3

obs = torch.rand((N,1,20,20))
action = torch.rand(N,3)
reward = torch.rand(N,1)

for _ in range(100):
    pred, = net(obs, (action,))
    loss = ((pred - reward)**2).sum()
    print(reward, pred, loss)
    net.optimise(loss)