import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('/media/meow/Transcend/Quantum_curve_predict/realtime_results.csv')

# Display basic information about the data
print("Original data shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nUnique pools:", df['pool_name'].unique())
print("\nUnique models:", df['model'].unique())

# Group by pool_name and model, then calculate mean for numeric columns
numeric_columns = ['test_mae', 'test_rmse', 'test_direction_acc', 'train_mae', 'train_rmse', 'train_direction_acc']

# Group and calculate averages by pool and model
grouped_df = df.groupby(['pool_name', 'model'])[numeric_columns].mean().reset_index()

# Round the numeric values to 6 decimal places for better readability
for col in numeric_columns:
    grouped_df[col] = grouped_df[col].round(6)

# Calculate model averages across all pools
model_averages = df.groupby('model')[numeric_columns].mean().reset_index()
model_averages['pool_name'] = 'ALL_POOLS_AVERAGE'  # Add a special identifier
model_averages = model_averages[['pool_name', 'model'] + numeric_columns]  # Reorder columns

# Round the model averages to 6 decimal places
for col in numeric_columns:
    model_averages[col] = model_averages[col].round(6)

# Combine the grouped data with model averages
final_df = pd.concat([grouped_df, model_averages], ignore_index=True)

# Display the results
print("\nProcessed data shape:", final_df.shape)
print("\nFirst few rows of processed data:")
print(final_df.head(10))

# Save to new CSV file
output_file = '/media/meow/Transcend/Quantum_curve_predict/averaged_results.csv'
final_df.to_csv(output_file, index=False)

print(f"\nAveraged results saved to: {output_file}")

# Display summary statistics
print("\nSummary by pool:")
for pool in grouped_df['pool_name'].unique():
    pool_data = grouped_df[grouped_df['pool_name'] == pool]
    print(f"\n{pool} ({len(pool_data)} models):")
    for _, row in pool_data.iterrows():
        print(f"  {row['model']}: Test MAE={row['test_mae']:.6f}, Test RMSE={row['test_rmse']:.6f}, Test Acc={row['test_direction_acc']:.6f}")

# Display model averages across all pools
print("\n" + "="*80)
print("MODEL AVERAGES ACROSS ALL POOLS:")
print("="*80)
for _, row in model_averages.iterrows():
    print(f"{row['model']}:")
    print(f"  Test MAE: {row['test_mae']:.6f}")
    print(f"  Test RMSE: {row['test_rmse']:.6f}")
    print(f"  Test Direction Accuracy: {row['test_direction_acc']:.6f}")
    print(f"  Train MAE: {row['train_mae']:.6f}")
    print(f"  Train RMSE: {row['train_rmse']:.6f}")
    print(f"  Train Direction Accuracy: {row['train_direction_acc']:.6f}")
    print() 