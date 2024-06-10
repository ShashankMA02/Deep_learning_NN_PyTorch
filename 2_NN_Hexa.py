import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %matplotlib inline

class FKModel(nn.Module):
    def __init__(self, in_features=6, hidden_size=12, num_layers=500, out_features=6):
        super(FKModel, self).__init__()
        
        # Initial input layer
        self.input_layer = nn.Linear(in_features, hidden_size)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.output_layer(x)
        return x

torch.manual_seed(42)
model = FKModel()
print(model)

motor_url = 'https://raw.githubusercontent.com/ShashankMA02/CSV_data_of_Hexa/main/XY_0_50_s1_mot.csv'
position_url = 'https://raw.githubusercontent.com/ShashankMA02/CSV_data_of_Hexa/main/XY_0_50_s1_pos.csv'

mot_df = pd.read_csv(motor_url)
pos_df = pd.read_csv(position_url)

# Train Test Split set X, y
X = mot_df  # Motors values as input
y = pos_df  # Position values as output , hence Forward Kinematics

# Normalize the data
scaler_motors = StandardScaler()
scaler_positions = StandardScaler()

X = scaler_motors.fit_transform(X)
y = scaler_positions.fit_transform(y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

# Convert y features to float tensors
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

# Criterion to measure error of prediction
criterion = nn.MSELoss()

# Optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Model training
epochs = 1000
losses = []

for i in range(epochs):
    # go forward and get a prediction
    y_pred = model.forward(X_train)      # get predicted results

    # Measure the loss/error, gonna be high at first
    loss = criterion(y_pred, y_train)    #predicted values vs the y_train

    # keep track of our losses
    losses.append(loss.detach().numpy())

    # print every 10 epochs
    if i % 10 == 0:
        print(f'Epochs: {i} and loss: {loss}')

    # Do some backpropagation: take the error rate of forward propagation and
    # feed it back through the network to fine-tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Graph it out
plt.plot(range(epochs), losses)
plt.ylabel('Loss / Error')
plt.xlabel('Epochs')
plt.show()

# Evaluate Model on Test data set
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = criterion(y_eval, y_test)
    print(f'Loss: {loss}')

# New motor values for prediction
motors_new = [[2215, 1881, 2215, 1879, 2217, 1881]]  # Replace with actual new motor values

# Normalize the new motor values
motors_new = scaler_motors.transform(motors_new)
motors_new = torch.tensor(motors_new, dtype=torch.float32)

# Make predictions
model.eval()
with torch.no_grad():
    positions_pred = model(motors_new)
    positions_pred = scaler_positions.inverse_transform(positions_pred.numpy())

print('Predicted positions:', positions_pred)
