import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #since pytorch doesn't have a built-in StandardScaler
from sklearn.metrics import r2_score #since pytorch doesn't have a built-in r2
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import copy
import matplotlib.pyplot as plt

#set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#import dataset of distillation column; contains 450 hours of operation at 0.1 hour samples
data = pd.read_csv("dataset_distill.csv", sep=";")
#print(data) #we see there are 14 trays in this column as well
#print()

#looking at the data, some rows have extreme outliers for L and V (for example 22500 vs 1.23*10^8)
#let's get these rows out, but first, check for missing values
#print(data.isnull().sum()) #no missing values... but we see the dtype is int64
#print()

#convert dtype to float64
#print(data.dtypes) #last few columns only are int64
data = data.astype("float64")
#print(data.dtypes)
#print()

#now to get those outliers outta here
#from the data, we see that the rows with L=1.23e8 also have V=1.23e8, so we'll get rid of any rows with L=1.23e8
mask = data["L"] > 1.21e8
cleaned_data = data.drop(data.loc[mask].index,axis=0)
#print(cleaned_data)
#print()

#get X and y arrays from data
X = cleaned_data.drop(columns=["Ethanol concentration"]).values #features
y = cleaned_data["Ethanol concentration"].values.reshape(-1,1) #targets
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=4)
#On another note, could split data based on time ranges or change test size proportion
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#convert arrays to tensors
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

train_set = TensorDataset(X_train_scaled, y_train)
val_test_set = TensorDataset(X_test_scaled, y_test)
val_set, test_set = random_split(val_test_set, [0.5,0.5], generator=torch.Generator().manual_seed(4)) #could change these proportions
#now, from the original data, 80% is train, 10% is val, 10% is test

def create_train_eval_model(intermediary, lr, optim_weight_decay, sch_factor, sch_patience, loader_batch_size, num_epochs, train_patience):
    '''
    Quick way of some modifying model parameters to help create the best model
    
    Will save the evaluation metrics in a dictionary for comparison with other models
    '''
    #create a 2-layer model with ReLU activation that will predict ethanol concentrations based on all the other measured data
    torch.manual_seed(4)
    
    model = nn.Sequential(
        nn.Linear(X.shape[1],intermediary),
        nn.ReLU(),
        nn.Linear(intermediary,1)
    ) #could change to LSTM or GRU
    model = model.to(device)
    
    #loss function, Adam optimizer, ReduceLROnPlateau scheduler and loaders for training and evaluation
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=optim_weight_decay) #weight decay is to prevent overfitting
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=sch_factor, patience=sch_patience)
    train_loader = DataLoader(train_set, batch_size=loader_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=loader_batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=loader_batch_size, shuffle=False)
    
    #train the model
    def average_loss(model, loader, loss_fn): #will be used to compare training loss with validation loss to check for overfitting
        model.eval()
        total = 0.0
        with torch.no_grad():
            for features, target in loader:
                preds = model(features) #preds is short for predictions
                total += loss_fn(preds, target).item()
        return total/len(loader)
    
    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    print("Epoch | Train_loss | Val_loss")
    for epoch in range(1,num_epochs+1): #train for (num_epochs) epochs
        model.train()
        train_total = 0.0
        for features, target in train_loader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            train_total += loss.item()
        train_loss = train_total / len(train_loader)
        val_loss = average_loss(model, val_loader, loss_fn)
        scheduler.step(val_loss)
        
        if epoch % 2 == 0: #print losses every 2 epochs
            print(epoch, "|", round(train_loss, 10), "|", round(val_loss, 10))
        #if the val_loss stops dropping while train_loss continues dropping, we have overfitting; if after (train_patience) epochs the val_losses stop dropping, stop training
        if val_loss < best_val:
            best_val = val_loss
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict()) #save the weights of the epoch with the smallest loss
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement == train_patience:
                break
    
    #load the model with the smallest loss
    model.load_state_dict(best_state)
    
    #evaluate the model
    concentrations = [] #we need to combine the batches of predictions from the model during eval below
    targets = [] #we need to get the targets of the test set as a tensor
    
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            concentrations.append(preds)
            targets.append(y)
    
    concentrations = torch.cat(concentrations).to('cpu')
    targets = torch.cat(targets).to('cpu')
    
    mse = loss_fn(concentrations, targets)
    r2 = r2_score(targets.numpy(), concentrations.numpy())
    mae = nn.L1Loss()(concentrations, targets)
    rmse = mse.item() ** (1/2)
    residuals = targets - concentrations
    
    print()
    print("Model Evaluation")
    print(f"MSE: {mse.item():.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae.item():.6f}")
    print(f"R²: {r2:.6f}")
    
    plt.figure()
    plt.scatter(targets, concentrations)
    plt.plot([targets.min(), targets.max()],[targets.min(), targets.max()],color="red",linestyle="--") #red dashed y=x line representing perfect fit
    plt.xlabel("Actual EtOH concentrations")
    plt.ylabel("Predicted EtOH concentrations")
    plt.title("Actual vs predicted concentrations scatter plot for 2-layer model")
    
    plt.figure()
    plt.scatter(concentrations, residuals)
    plt.axhline(y=0, color="red",linestyle="--") #red dashed y=0 line representing perfect fit
    plt.title("Residual plot for soft sensor model")
    plt.xlabel("Predicted EtOH concentrations")
    plt.ylabel("Residuals")
    plt.show()
    
    return {"MSE": mse.item(), "RMSE": rmse, "MAE": mae.item(), "R²": r2, "val_loss": best_val}, model

#test parameters to create the best model
attempt1 = create_train_eval_model(40,0.1,1e-4,0.5,5,10,100,5)
print()
attempt2 = create_train_eval_model(40,0.1,1e-4,0.5,5,10,100,10)
print()
attempt3 = create_train_eval_model(40,0.01,1e-4,0.5,5,10,100,10)
print()
attempt4, model4 = create_train_eval_model(40,0.001,1e-4,0.5,5,10,100,10) #best of the 4 attempts so far
#could continue to experiment, but would probably need a bigger dataset for further model evaluation