import torch
from networks import ResNet18, MultiHeadedNetwork


net = MultiHeadedNetwork(
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    common = ResNet18(in_channels=1),
    head_codes = [[(131, 1)]], # 128), "R", (128, 1)),),
    lr=1e-5,
)

obs = torch.rand((2,1,2,2))
action = torch.rand(2,3)
reward = torch.tensor([1,2])

for _ in range(1000):
    pred, = net(obs, (action,))
    print(pred)
    loss = ((pred - reward)**2).sum()
    net.optimise(loss)