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

val_frac = 0.2
n_train = int((1-val_frac) * len(X_train))

X_val = X_train[n_train:]
y_val = y_train[n_train:]

X_train = X_train[:n_train]
y_train = y_train[:n_train]

X_test, y_test = generate_data(1000)

# 2. Calculate true conditional mean

def true_conditional_mean(x):
    w = torch.tensor([4.0, 0.0])
    b = 0.0
    logits = x @ w + b
    return torch.sigmoid(logits)

y_val_bayes = (true_conditional_mean(X_val) > 0.5).long()
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
    teacher_logits_train = teacher(X_train)
    teacher_logits_val = teacher(X_val)

    T = 4

    teacher_soft_train = F.softmax(teacher_logits_train / T, dim=1)
    teacher_soft_val = F.softmax(teacher_logits_val / T, dim = 1)

    teacher_hard_train = teacher_soft_train.argmax(dim = 1)
    teacher_hard_val = teacher_soft_val.argmax(dim = 1)

# Step 5. Train Student model

def train_student(target_type):
    student = Student().to(device)
    opt = optim.Adam(student.parameters(), lr = 1e-3)

    train_losses = []
    val_losses = []

    for epoch in range(500):
        student.train()
        opt.zero_grad()
        logits = student(X_train)

        if target_type == "teacher_hard":
            loss = F.cross_entropy(logits, teacher_hard_train)
        
        elif target_type == "bayes_hard":
            loss = F.cross_entropy(logits, y_train_bayes)
        
        elif target_type == "teacher_soft":
            log_probs = F.log_softmax(logits, dim = 1)
            loss = F.kl_div(log_probs, teacher_soft_train, reduction = "batchmean")

        loss.backward()
        opt.step()

        train_losses.append(loss.item())

        student.eval()
        with torch.no_grad():
            val_logits = student(X_val)

            if target_type == "teacher_hard":
                val_loss = F.cross_entropy(val_logits, teacher_hard_val)
            
            elif target_type == "bayes_hard":
                val_loss = F.cross_entropy(val_logits, y_val_bayes)
            
            elif target_type == "teacher_soft":
                val_log_probs = F.log_softmax(val_logits / T, dim = 1)
                val_loss = F.kl_div(val_log_probs, teacher_soft_val, reduction = "batchmean")

        val_losses.append(val_loss.item())
    
    return student, train_losses, val_losses

# Case 1: Hard labels from teacher output
model_a, train_a, val_a = train_student("teacher_hard")

# Case 2: Use hard labels but averaged from the true conditional mean
model_b, train_b, val_b = train_student("bayes_hard")

# Case 3: Soft label
model_c, train_c, val_c = train_student("teacher_soft")

plt.figure(figsize=(8,5))

plt.plot(train_a, label="Train - Hard Labels")
plt.plot(val_a, '--', label="Val - Hard Labels")

plt.plot(train_b, label="Train - Conditional Mean of Hard Labels")
plt.plot(val_b, '--', label="Val - Conditional Mean of Hard Labels")

plt.plot(train_c, label="Train - Soft Labels")
plt.plot(val_c, '--', label="Val - Soft Labels")

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
