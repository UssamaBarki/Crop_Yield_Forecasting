#%% md
# ### Submitted by Candidate : 291767
#%% md
# # 1- Data Merging and Cleaning
#%%
# ===================== Data Merging ===================== #

# Import necessary libraries
import pandas as pd
import numpy as np
import math

# Monthly to Anual data conversion
def aggregate_monthly_data(df):
    """
    Compute the annual mean from columns 2 to 13 (assumed to be monthly data).
    This version uses fixed column positions, making it general for similarly structured data.
    """
    df['annual_mean'] = df.iloc[:, 2:14].mean(axis=1)
    return df

# Half grid adjustment for merging
def force_to_half_grid(x):
    """
    Snaps the coordinate x to the fixed grid with centers at n + 0.5.
    """
    if pd.isnull(x):
        return x
    return math.floor(x) + 0.5

# Load Yield CSV
df = pd.read_csv('Yield_and_Production_data.csv')

# Pivot so Yield and Production become separate columns
yield_df = (
    df
    .pivot_table(
        index=['Country', 'Item Code (CPC)', 'Item', 'Year'],
        columns='Element',
        values='Value'
    )
    .reset_index()
)

yield_df.drop(['Item Code (CPC)'], axis=1, inplace=True)

# Load lookup CSV file
lookup_df = pd.read_csv('country_latitude_longitude_area_lookup.csv')

# Drop rows where all values are NaN
lookup_df = lookup_df.drop(lookup_df.index[227])

# Half grid application
lookup_df['latitude'] = lookup_df['centroid latitude'].apply(force_to_half_grid)
lookup_df['longitude'] = lookup_df['centroid longitude'].apply(force_to_half_grid)

# Remove the columns 'area' and 'centroid radius' from df_lookup
lookup_df.drop(['area', 'centroid radius','centroid latitude','centroid longitude'], axis=1, inplace=True)

# Rename column in lookup_df to match yield_df's column.
lookup_df.rename(columns={'country': 'Country'}, inplace=True)

# Merge on the common key "Country"
df_merged = yield_df.merge(lookup_df, on='Country', how='left')

# Optionally, rename coordinates for consistency.
df_merged.rename(columns={'Year': 'year'}, inplace=True)

### Monthly to yearly aggregate ####

#1 Load the canopy dataset and compute its annual statistic.
canopy_df = pd.read_csv('CanopInt_inst_data.csv')
canopy_df = aggregate_monthly_data(canopy_df)
# Rename the aggregated column to reflect its content
canopy_df.rename(columns={'annual_mean': 'Canopy_annual_mean'}, inplace=True)

#2 Load the rainfall dataset and aggregate monthly values.
rain_df = pd.read_csv('Rainf_tavg_data.csv')
rain_df = aggregate_monthly_data(rain_df)
rain_df.rename(columns={'annual_mean': 'Rainfall_annual_mean'}, inplace=True)

#3 Load the snow fall average dataset aggregate monthly values.
snow_df = pd.read_csv('Snowf_tavg_data.csv')
snow_df = aggregate_monthly_data(snow_df)
snow_df.rename(columns={'annual_mean': 'Snowfall_annual_mean'}, inplace=True)

#4 Load the transpiration dataset aggregate monthly values.
transpiration_df = pd.read_csv('TVeg_tavg_data.csv')
transpiration_df = aggregate_monthly_data(transpiration_df)
transpiration_df.rename(columns={'annual_mean': 'Transpiration_annual_mean'}, inplace=True)

#5 Load the evaporation from soil aggregate monthly values.
evaporation_df = pd.read_csv('ESoil_tavg_data.csv')
evaporation_df = aggregate_monthly_data(evaporation_df)
evaporation_df.rename(columns={'annual_mean': 'Evaporation_annual_mean'}, inplace=True)

#6 Load the terrestrial water storage aggregate monthly values.
water_storage_df = pd.read_csv('TWS_inst_data.csv')
water_storage_df = aggregate_monthly_data(water_storage_df)
water_storage_df.rename(columns={'annual_mean': 'Terrestrial_water_storage_annual_mean'}, inplace=True)


#7 Load the soil moisture dataset for 0-10 cm and aggregate.
soil_moisture_0_10_df = pd.read_csv('SoilMoi0_10cm_inst_data.csv')
soil_moisture_0_10_df = aggregate_monthly_data(soil_moisture_0_10_df)
soil_moisture_0_10_df.rename(columns={'annual_mean': 'SoilMoisture0_10_annual_mean'}, inplace=True)

#8 Load the soil moisture dataset for 10-40 cm and aggregate.
soil_moisture_10_40_df = pd.read_csv('SoilMoi10_40cm_inst_data.csv')
soil_moisture_10_40_df = aggregate_monthly_data(soil_moisture_10_40_df)
soil_moisture_10_40_df.rename(columns={'annual_mean': 'SoilMoisture10_40_annual_mean'}, inplace=True)

#9 Load the soil moisture dataset for 40-100 cm and aggregate.
soil_moisture_40_100_df = pd.read_csv('SoilMoi40_100cm_inst_data.csv')
soil_moisture_40_100_df = aggregate_monthly_data(soil_moisture_40_100_df)
soil_moisture_40_100_df.rename(columns={'annual_mean': 'SoilMoisture40_100_annual_mean'}, inplace=True)

#10 Load the soil moisture dataset for 100-200 cm and aggregate.
soil_moisture_100_200_df = pd.read_csv('SoilMoi100_200cm_inst_data.csv')
soil_moisture_100_200_df = aggregate_monthly_data(soil_moisture_100_200_df)
soil_moisture_100_200_df.rename(columns={'annual_mean': 'SoilMoisture100_200_annual_mean'}, inplace=True)

#11 Load the soil temperature dataset for 0-10 cm and aggregate.
soil_temperature_0_10_df = pd.read_csv('SoilTMP0_10cm_inst_data.csv')
soil_temperature_0_10_df = aggregate_monthly_data(soil_temperature_0_10_df)
soil_temperature_0_10_df.rename(columns={'annual_mean': 'SoilTemperature0_10_annual_mean'}, inplace=True)

#12 Load the soil temperature dataset for 10-40 cm and aggregate.
soil_temperature_10_40_df = pd.read_csv('SoilTMP10_40cm_inst_data.csv')
soil_temperature_10_40_df = aggregate_monthly_data(soil_temperature_10_40_df)
soil_temperature_10_40_df.rename(columns={'annual_mean': 'SoilTemperature10_40_annual_mean'}, inplace=True)

#13 Load the soil temperature dataset for 40-100 cm and aggregate.
soil_temperature_40_100_df = pd.read_csv('SoilTMP40_100cm_inst_data.csv')
soil_temperature_40_100_df = aggregate_monthly_data(soil_temperature_40_100_df)
soil_temperature_40_100_df.rename(columns={'annual_mean': 'SoilTemperature40_100_annual_mean'}, inplace=True)

#14 Load the soil temperature dataset for 100-200 cm and aggregate.
soil_temperature_100_200_df = pd.read_csv('SoilTMP100_200cm_inst_data.csv')
soil_temperature_100_200_df = aggregate_monthly_data(soil_temperature_100_200_df)
soil_temperature_100_200_df.rename(columns={'annual_mean': 'SoilTemperature100_200_annual_mean'}, inplace=True)

#15 left "Land_cover_percent_data.csv"' aggregate as it doesnot have any monthly data, instead have classes    
land_cover_df = pd.read_csv('Land_cover_percent_data.csv') 

# 16,17(yield and lookup) merged already in df_merged. 


### Merging ###

# Merge the yield data (which now includes Latitude, Longitude, Year, and Country) with the monthly datasets.
# We use only the columns we need from each monthly dataset.

df_merged = df_merged.merge(canopy_df[['longitude', 'latitude', 'year', 'Canopy_annual_mean']],
                           on=['longitude', 'latitude', 'year'], how='left')

#2 Merge with rainfall data
df_merged = df_merged.merge(rain_df[['longitude', 'latitude', 'year', 'Rainfall_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')

#3 Merge with snow fall data
df_merged = df_merged.merge(snow_df[['longitude', 'latitude', 'year', 'Snowfall_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#4 Merge with transpiration data.
df_merged = df_merged.merge(transpiration_df[['longitude', 'latitude', 'year', 'Transpiration_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#5 Merge with evaporation data.
df_merged = df_merged.merge(evaporation_df[['longitude', 'latitude', 'year', 'Evaporation_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#6 Merge with terrestrial water data.
df_merged = df_merged.merge(water_storage_df[['longitude', 'latitude', 'year', 'Terrestrial_water_storage_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#7 # Merge with soil moisture (0-10cm) data
df_merged = df_merged.merge(soil_moisture_0_10_df[['longitude', 'latitude', 'year', 'SoilMoisture0_10_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#8 Merge with soil moisture 10-40 cm data.
df_merged = df_merged.merge(soil_moisture_10_40_df[['longitude', 'latitude', 'year', 'SoilMoisture10_40_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#9 Merge with soil moisture 40-100 cm data.
df_merged = df_merged.merge(soil_moisture_40_100_df[['longitude', 'latitude', 'year', 'SoilMoisture40_100_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#10 Merge with soil moisture 100-200 cm data.
df_merged = df_merged.merge(soil_moisture_100_200_df[['longitude', 'latitude', 'year', 'SoilMoisture100_200_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#11 Merge with soil temperature 0-10 cm data.
df_merged = df_merged.merge(soil_temperature_0_10_df[['longitude', 'latitude', 'year', 'SoilTemperature0_10_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#12 Merge with soil temperature 10-40 cm data.
df_merged = df_merged.merge(soil_temperature_10_40_df[['longitude', 'latitude', 'year', 'SoilTemperature10_40_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#13 Merge with soil temperature 40-100 cm data.
df_merged = df_merged.merge(soil_temperature_40_100_df[['longitude', 'latitude', 'year', 'SoilTemperature40_100_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#14 Merge with soil temperature 100-200 cm data.
df_merged = df_merged.merge(soil_temperature_100_200_df[['longitude', 'latitude', 'year', 'SoilTemperature100_200_annual_mean']],
                            on=['longitude', 'latitude', 'year'], how='left')


#15 Merge with country_latitude_longitude_area_lookup.csv", 
# already merged in yield file 

#16 Merge the land cover dataset with the main merged dataset based on matching keys
df_merged = df_merged.merge(land_cover_df, on=['longitude', 'latitude', 'year'], how='left')


# Rename columns
df_merged.rename(columns={'year': 'Year','longitude':'Longitude','latitude':'Latitude'}, inplace=True)


# ===================== Data Cleaning ===================== #


# renaming of column
df_merged.rename(columns={'Land_cover_percent_classh_4': 'Land_cover_percent_class_4'}, inplace=True)

# Dropping longitude and latitude:
columns_to_drop = ['Longitude','Latitude']  # adjust as necessary
df_merged.drop(columns=columns_to_drop, errors='ignore', inplace=True)


# Drop Null Values in Yield
df_clean = df_merged.dropna(subset=['Yield'])

# Drop Null values in Cannopy_annual_mean
df_clean = df_clean.dropna(subset=['Canopy_annual_mean'])


# Reorder the columns:
new_order = ['Country','Year','Item','Canopy_annual_mean', 'Rainfall_annual_mean', 'Snowfall_annual_mean', 
                     'Transpiration_annual_mean', 'Evaporation_annual_mean', 
                     'Terrestrial_water_storage_annual_mean', 'SoilMoisture0_10_annual_mean', 
                     'SoilMoisture10_40_annual_mean', 'SoilMoisture40_100_annual_mean', 
                     'SoilMoisture100_200_annual_mean', 'SoilTemperature0_10_annual_mean', 
                     'SoilTemperature10_40_annual_mean', 'SoilTemperature40_100_annual_mean', 
                     'SoilTemperature100_200_annual_mean','Land_cover_percent_class_1',
                     'Land_cover_percent_class_2','Land_cover_percent_class_3','Land_cover_percent_class_4',
                     'Land_cover_percent_class_5','Land_cover_percent_class_6','Land_cover_percent_class_7',
                     'Land_cover_percent_class_8','Land_cover_percent_class_9','Land_cover_percent_class_10',
                     'Land_cover_percent_class_11','Land_cover_percent_class_12','Land_cover_percent_class_13',
                     'Land_cover_percent_class_14','Land_cover_percent_class_15','Land_cover_percent_class_16',
                     'Land_cover_percent_class_17','Production','Yield']

# Reorder the DataFrame
df_clean = df_clean[new_order]

#%% md
# # 2 - MLP Model
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.stats import mstats, spearmanr

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ===================== Define Model and Trainer ===================== #

class EnhancedMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.block2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.output = nn.Linear(128, 1)

         # Residual connections to help gradient flow
        self.res1 = nn.Linear(input_size, 512) if input_size != 512 else nn.Identity()
        self.res2 = nn.Linear(512, 256) if input_size != 256 else nn.Identity()

    def forward(self, x):
        identity1 = self.res1(x)
        x = self.block1(x) + identity1
        
        identity2 = self.res2(x)
        x = self.block2(x) + identity2
        
        x = self.block3(x)
        return self.output(x).squeeze()


class AdvancedTrainer:
    def __init__(self, model, patience=20, val_frac=0.15):
        self.model = model
        self.patience = patience
        self.val_frac = val_frac
        self.train_losses = []
        self.val_mae = []
        self.best_epoch = 0
        self.best_model_state = None

    def fit(self, X, y, epochs=100, lr=0.001):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.val_frac, random_state=42
        )
        
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Custom loss function: 70% Huber + 30% L1
        def custom_loss(pred, target):
            huber = nn.HuberLoss()(pred, target)
            return 0.7 * huber + 0.3 * nn.L1Loss()(pred, target)
            
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr, 
            steps_per_epoch=len(train_loader),
            epochs=epochs
        )
        
        best_mae = np.inf
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = custom_loss(outputs, y_batch)
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), 0.5)
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(torch.tensor(X_val, dtype=torch.float32))
                val_mae_calc = mean_absolute_error(y_val, val_pred.numpy())
                self.val_mae.append(val_mae_calc)
                self.train_losses.append(epoch_loss/len(train_loader))
            
            # Early stopping
            if val_mae_calc < best_mae:
                best_mae = val_mae_calc
                self.best_epoch = epoch
                epochs_no_improve = 0
                self.best_model_state = self.model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

def tta_predict(model, X, n=10):
    # Enable dropout at test time for uncertainty estimation
    model.train()
    with torch.no_grad():
        preds = [model(torch.tensor(X, dtype=torch.float32)) for _ in range(n)]
    return torch.mean(torch.stack(preds), 0)

# ===================== Data Loading and Preprocessing ===================== #
# Read and sort data
df = df_clean
df = df.sort_values(['Country', 'Item', 'Year'])

# Feature engineering with temporal patterns
df['Next_Year_Yield'] = df.groupby(['Country', 'Item'])['Yield'].shift(-1)
for lag in [1, 2, 3]:
    df[f'Yield_Lag_{lag}'] = df.groupby(['Country', 'Item'])['Yield'].shift(lag)
df['3Yr_Avg_Yield'] = df.groupby(['Country', 'Item'])['Yield'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
df_shifted = df.dropna(subset=['Next_Year_Yield'])

# Handle outliers in target via winsorization
y = df_shifted['Next_Year_Yield']
y = pd.Series(mstats.winsorize(y, limits=[0.01, 0.01]), index=y.index)

# Prepare features
X = df_shifted.drop(['Next_Year_Yield', 'Yield', 'Production'], axis=1)

# Temporal split: train using data before 2021 and test on 2021 and later
train_mask = df_shifted['Year'] < 2021
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

# Target scaling
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

# Preprocess features:
# - Use PowerTransformer for numeric features.
# - Encode categorical features ('Country', 'Item') with OrdinalEncoder.
categorical_features = ['Country', 'Item']
numeric_features = [col for col in X.columns if col not in 
                    ['Country', 'Item', 'Year'] + [f'Yield_Lag_{i}' for i in [1,2,3]]]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', PowerTransformer(), numeric_features),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ===================== Model Training ===================== #

input_size = X_train_processed.shape[1]
model = EnhancedMLP(input_size)
trainer = AdvancedTrainer(model, patience=20)
print("Training model...")
trainer.fit(X_train_processed, y_train_scaled, epochs=100, lr=0.001)

# Enhanced prediction with Test Time Augmentation (TTA)
y_pred_scaled = tta_predict(model, X_test_processed).numpy()
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# Calculate metrics on the original yield scale
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
huber_fn = nn.HuberLoss()
r2 = r2_score(y_test, y_pred)
pearson_corr = stats.pearsonr(y_test, y_pred)[0]

# Compute Huber Loss on the original scale (convert numpy to torch tensors)
huber_loss_val = huber_fn(torch.tensor(y_pred, dtype=torch.float32), 
                            torch.tensor(y_test.values, dtype=torch.float32)).item()
spearman_corr, _ = spearmanr(y_test, y_pred)

print(f"\nBest Epoch: {trainer.best_epoch}")
print("Test Regression Metrics:")
print(f"- MAE: {mae:.2f}")
print(f"- MSE: {mse:.2f}")
print(f"- RMSE: {rmse:.2f}")
print(f"- Huber Loss: {huber_loss_val:.2f}")
print(f"- R²: {r2:.2f}")
print(f"- Pearson's r: {pearson_corr:.2f}")
print(f"- Spearman's r: {spearman_corr:.2f}")


# ===================== Plotting ===================== #

# Plot training curves
plt.figure(figsize=(12, 6))
plt.plot(trainer.train_losses, label='Training Loss')
plt.plot(trainer.val_mae, label='Validation MAE')
plt.axvline(trainer.best_epoch, color='r', linestyle='--', label='Best Epoch')
plt.title('Training Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss/MAE')
plt.legend()
plt.grid(True)
#plt.savefig('Plot training curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Actual vs Predicted Yield scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Yield - Kg/ha')
plt.ylabel('Predicted Yield - Kg/ha')
plt.title('Actual vs Predicted Yield')
plt.grid(True)
#plt.savefig('Actual vs Predicted Plot.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== Generate Output Files ===================== #

# 2022 Predictions vs. Actuals (shift year by +1 for prediction comparison)
df_2022_comparison = X_test[['Country', 'Item', 'Year']].copy()
df_2022_comparison['Year'] += 1
df_2022_comparison['Actual Yield - Kg/ha'] = y_test.values
df_2022_comparison['Predicted Yield - Kg/ha'] = y_pred
df_2022_comparison.to_csv('2022_predictions_vs_actuals.csv', index=False)
print('Saved 2022 predictions vs actuals.csv')

# 2023 Predictions using data from 2022
X_2023 = df[df['Year'] == 2022].drop(['Yield', 'Production'], axis=1)
if not X_2023.empty:
    X_2023_processed = preprocessor.transform(X_2023)
    with torch.no_grad():
        y_2023_scaled = tta_predict(model, X_2023_processed).numpy()
    y_2023 = y_scaler.inverse_transform(y_2023_scaled.reshape(-1, 1)).flatten()
    predictions_2023 = X_2023[['Country', 'Item']].copy()
    predictions_2023['Year'] = 2023
    predictions_2023['Predicted Yield - Kg/ha'] = y_2023
    predictions_2023.to_csv('2023_predictions.csv', index=False)
    print('Saved 2023 predictions.csv')
else:
    print("\nNo data available for 2023 predictions")

# Combined results: merging 2022 and 2023 prediction outputs
results = (
    X_test[['Country', 'Item', 'Year']]
    .assign(Year=lambda x: x.Year + 1,
            Actual_Yield_2022=y_test.values,
            Predicted_Yield_2022=y_pred,
            Error_Pct=((y_pred - y_test)/y_test*100).round(2))
    .sort_values(['Country', 'Item'])
)
if not X_2023.empty:
    results = results.merge(
        predictions_2023[['Country', 'Item', 'Predicted Yield - Kg/ha']].rename(columns={'Predicted Yield - Kg/ha': 'Predicted Yield 2023 - Kg/ha'}),
        on=['Country', 'Item'], how='left'
    )

results.to_csv('yield_predictions_combined_2022_2023.csv', index=False)
print('Saved yield predictions combined.csv')

#%%
