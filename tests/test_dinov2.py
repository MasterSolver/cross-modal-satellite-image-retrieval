from models.dinov2 import load_dinov2

model, device = load_dinov2()

print(device)
print(next(model.parameters()).device)