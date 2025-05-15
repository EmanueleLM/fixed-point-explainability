import torch

def train(model, loader, criterion, optimizer, device="cpu"):
    model.train()
    total_loss = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Train Loss: {total_loss / len(loader):.4f}")

# === Evaluation function ===
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


# === Evaluation function ===
def evaluate_per_class(model, loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    accuracy_per_class = {f"{i}": (0, 0) for i in range(10)}  # Assuming 10 classes
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for l,c in zip(labels, predicted):
                if l.item() == c.item():
                    accuracy_per_class[str(l.item())] = accuracy_per_class[str(l.item())][0] + 1, accuracy_per_class[str(l.item())][1] + 1
                else:
                    accuracy_per_class[str(l.item())] = accuracy_per_class[str(l.item())][0], accuracy_per_class[str(l.item())][1] + 1

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    for k, v in accuracy_per_class.items():
        if v[1] == 0:
            accuracy_per_class[k] = 0
        else:
            accuracy_per_class[k] = v[0] / v[1] * 100
    
    return accuracy, accuracy_per_class