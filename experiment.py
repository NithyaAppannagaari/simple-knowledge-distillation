import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

device = "cpu"

# 1. Create a synthetic dataset

def generate_data(n = 2000):
    mu0 = torch.tensor([-2, 0.0])
    mu1 = torch.tensor([2, 0.0])

    x0 = torch.randn(n//2, 2) + mu0
    x1 = torch.randn(n//2, 2) + mu1

    X = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n//2, dtype = torch.long),
                   torch.ones(n//2, dtype = torch.long)])
    
    return X, y

X_train, y_train = generate_data(2000)
X_test, y_test = generate_data(1000)

# 2. Calculate true conditional mean

def true_conditional_mean(x):
    w = torch.tensor([4.0, 0.0])
    b = 0.0
    logits = x @ w + b
    return torch.sigmoid(logits)

y_train_bayes = (true_conditional_mean(X_train) > 0.5).long()

class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.net(x)

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )
    
    def forward(self, x):
        return self.net(x)

# Step 3. Train Teacher model

teacher = Teacher().to(device)
opt = optim.Adam(teacher.parameters(), lr = 1e-3)

for _ in range(500):
    opt.zero_grad()
    logits = teacher(X_train)
    loss = F.cross_entropy(logits, y_train)
    loss.backward()
    opt.step()

teacher.eval()

# Step 4. Obtain distillation targets

with torch.no_grad():
    teacher_logits = teacher(X_train)
    T = 4
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    teacher_hard = teacher_soft.argmax(dim = 1)

# Step 5. Train Student model

def train_student(target_type):
    student = Student().to(device)
    opt = optim.Adam(student.parameters(), lr = 1e-3)

    losses = []
    accuracies = []

    for epoch in range(500):
        opt.zero_grad()
        logits = student(X_train)

        if target_type == "teacher_hard":
            loss = F.cross_entropy(logits, teacher_hard)
        
        elif target_type == "bayes_hard":
            loss = F.cross_entropy(logits, y_train_bayes)
        
        elif target_type == "teacher_soft":
            log_probs = F.log_softmax(logits, dim = 1)
            loss = F.kl_div(log_probs, teacher_soft, reduction = "batchmean")

        loss.backward()
        opt.step()

        losses.append(loss.item())

        # calculate accuracy
        with torch.no_grad():
            preds = logits.argmax(dim = 1)
            acc = (preds == y_train).float().mean().item()
            accuracies.append(acc)
    
    return student, losses, accuracies

# Case 1: Hard labels from teacher output
model_a, loss_a, acc_a = train_student("teacher_hard")

# Case 2: Use hard labels but averaged from the true conditional mean
model_b, loss_b, acc_b = train_student("bayes_hard")

# Case 3: Soft label
model_c, loss_c, acc_c = train_student("teacher_soft")

# Step 6. Evaluate
def evaluate(model):
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim = 1)
        acc = (preds == y_test).float().mean()
    return acc.item()

plt.figure()
plt.plot(loss_a, label="Hard Labels")
plt.plot(loss_b, label="Conditional Mean of Hard Labels")
plt.plot(loss_c, label="Soft Labels")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(acc_a, label="Hard Labels")
plt.plot(acc_b, label="Conditional Mean of Hard Labels")
plt.plot(acc_c, label="Soft Labels")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()