from omnimind import OmniMind

# Initialize the system
system = OmniMind()

# Generate a response
prompt = "What is artificial consciousness?"
response = system.generate_response(prompt)
print(f"Response: {response}")

# Process multimodal input
text = "Analyze this image"
image_path = "examples/test_image.jpg"
result = system.process_multimodal_input(text=text, image_path=image_path)
print(f"Multimodal analysis: {result}")

# Generate neural network
requirements = {
    'input_size': 10,
    'output_size': 2,
    'hidden_layers': [64, 32]
}
network = system.generate_neural_network(requirements)

# Train network
data = {'x': [...], 'y': [...]}  # Your training data
system.train_neural_network(network, data)

# Make a conscious decision
situation = "Should I optimize for speed or accuracy?"
options = ["speed", "accuracy"]
decision = system.make_conscious_decision(situation, options)
print(f"Decision: {decision}")

# Analyze system state
status = system.analyze_self()
print(f"System status: {status}")

# Evolve architecture
system.evolve_architecture()
print("System evolved")
