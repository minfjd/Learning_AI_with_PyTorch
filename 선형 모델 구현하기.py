import torch                      # 텐서 연산 및 자동 미분
import torch.nn as nn             # 신경망 레이어, 손실 함수
import torch.optim as optim       # 최적화 알고리즘 (SGD 등)
import matplotlib.pyplot as plt   # 시각화

# ============================================================
# 1. 데이터 생성
# ============================================================
# shape: X (100, 2), y (100, )
torch.manual_seed(0)              # 재현성을 위해 난수 시드 고정
X = torch.randn(100, 2)           # 표준정규분포에서 2차원 점 100개 샘플링
# 정답 라벨: x1 + x2 > 0 이면 1, 아니면 0
# → 직선 x1 + x2 = 0 을 경계로 나뉘는 선형 분리 가능한 이진 분류 문제
y = (X[:, 0] + X[:, 1] > 0).long()

# ============================================================
# 2. 모델 정의 (로지스틱 회귀)
# ============================================================
model = nn.Sequential(
    nn.Linear(2, 1),   # z = w1*x1 + w2*x2 + b, shape: (batch, 1)
    nn.Sigmoid()       # z를 0~1 사이 확률값으로 변환 (클래스 1일 확률)
)

# 손실 함수: 이진 교차 엔트로피 (Sigmoid 출력 확률과 0/1 라벨 비교)
criterion = nn.BCELoss()
# 옵티마이저: 확률적 경사 하강법, 학습률 0.1
optimizer = optim.SGD(model.parameters(), lr=0.1)

loss_history = []  # epoch별 손실값 기록용

# ============================================================
# 3. 학습
# ============================================================
# 전체 데이터(100개)를 한 번에 넣는 full-batch 학습을 50회 반복
for epoch in range(50):
    optimizer.zero_grad()                # 이전 step의 기울기 초기화 (누적 방지)
    output = model(X).squeeze()         # 순전파: (100, 1) → (100, )
    loss = criterion(output, y.float()) # BCELoss는 float 타입 타깃이 필요
    loss.backward()                      # 역전파: 각 파라미터의 기울기 계산
    optimizer.step()                     # 기울기 방향으로 파라미터 업데이트
    loss_history.append(loss.item())     # 텐서 → 파이썬 숫자로 변환해 저장

# ============================================================
# 4. 손실 곡선 시각화
# ============================================================
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# ============================================================
# 5. 분류 결과 시각화
# ============================================================
# 추론 시에는 기울기 계산이 필요 없으므로 no_grad로 메모리/연산 절약
with torch.no_grad():
    # 확률이 0.5보다 크면 클래스 1, 아니면 클래스 0으로 예측
    preds = (model(X).squeeze() > 0.5).long()

# 예측된 클래스에 따라 색을 달리해 산점도 출력 (파랑=0, 빨강=1)
plt.scatter(X[:, 0], X[:, 1], c=preds, cmap='coolwarm')
plt.title("Classification Result")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()
