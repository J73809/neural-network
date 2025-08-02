from scripts import NeuralNetwork
from scripts import accuracy, cross_entropy
from scripts import generate_data

network = NeuralNetwork([7, 10, 10, 10, 10, 5])

learning_rate = 0.01
decay_rate = 0.99
steps = 100
epochs = 2000
training_data = generate_data(500)

for epoch in range(epochs):
    if epoch > 0 and epoch % steps == 0:
        learning_rate *= decay_rate
    
    total_loss = 0
    total_correct = 0
    
    for x, y in training_data:
        output, activations, zs = network.forward(x)
        
        loss = cross_entropy(output, y)
        total_loss += loss
        
        total_correct += accuracy(output, y)
        
        network.backward(x, y, activations, zs, learning_rate)
    
    if (epoch + 1) % 50 == 0:
        average_loss = total_loss / len(training_data)
        average_acc = total_correct / len(training_data) * 100
        print(f"Epoch: {epoch+1}/{epochs} - Loss: {average_loss:.4f} - Accuracy: {average_acc:.4f}%")
