import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %matplotlib inline

class FKModel(nn.Module):

    ## Input layer (6 motor values) --> Hidden layer1 (ex-12 neurons) --> Hidden layer2 (ex-18 neurons) --> Output layer (6 pos values)

    def __init__(self, in_features=6, h1=12, h2=18, out_features=6):
        super().__init__()  # instantiate our nn.module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
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
motors_new = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]  # Replace with actual new motor values

# Normalize the new motor values
motors_new = scaler_motors.transform(motors_new)
motors_new = torch.tensor(motors_new, dtype=torch.float32)

# Make predictions
model.eval()
with torch.no_grad():
    positions_pred = model(motors_new)
    positions_pred = scaler_positions.inverse_transform(positions_pred.numpy())

print('Predicted positions:', positions_pred)
