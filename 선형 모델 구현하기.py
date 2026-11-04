import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 데이터 생성
# shape: X (100, 2), y(100, )
torch.manual_seed(0)
X = torch.randn(100, 2)
y = (X[ :, 0] + X[:,1] > 0).long()

# 모델 정의
model = nn.Sequential(
    nn.Linear(2,1),    # shape: (batch, 1)
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

loss_history = []

#학습
for epoch in range(50) :
    optimizer.zero_grad()
    ootput = model.sqeeze() # shape : (100, )
    loss = criterion(output, y.float())
    loss.backward()
    optimizer.step()
    loss_history.append(loss.time())

# 손실 곡선 시각화
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# 분류 결과 시각화
with torch.no_grad() :
    preds = (model(X).squeeze() > 0.5).long()

plt.scatter(X[:, 0], X[:, 1], c=preds, cmap= 'coolwarm')
plt.title("Classification Result")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()








