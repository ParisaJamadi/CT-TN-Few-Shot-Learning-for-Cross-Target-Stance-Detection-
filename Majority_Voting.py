import random
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

items = [0, 1]
stance_map_dict = {
    'AGAINST': 0,
    'FAVOR': 1
}
#After finishing the experiments and saving the results for all shots we changed the target_pair
target_pair='Biden_Trump'
class_names = list(stance_map_dict.keys())

# Define the list of seed numbers
seed_numbers = [24, 524, 1024, 1524, 2024]

# Define the list of shot numbers. 
#We start with 100 and after completing the code run for it we change the shot_numbers to 200. We will do the process until 400 shot
shot_numbers = [100]
shot=100

# Define combinations of prediction lists with corresponding names
prediction_combinations = [
    {'name': 'like preds', 'lists': ['like-gr-ta']},
    {'name': 'follower preds', 'lists': ['follower-gr-ta']},
    {'name': 'friend preds', 'lists': ['friend-gr-ta']},
    {'name': 'tweet preds', 'lists': ['tw-ta']},
    {'name': 'all preds', 'lists': ['follower-gr-ta', 'like-gr-ta', 'friend-gr-ta', 'tw-ta']},
    {'name': 'friend+tweet preds', 'lists': ['friend-gr-ta', 'tw-ta']},
    {'name': 'follower+tweet preds', 'lists': ['follower-gr-ta', 'tw-ta']},
    {'name': 'like+tweet preds', 'lists': ['like-gr-ta', 'tw-ta']},
    {'name': 'follower+friend+like', 'lists': ['like-gr-ta', 'friend-gr-ta','follower-gr-ta']},
]
# Create an empty DataFrame with the desired rows and columns
row_names = [comb['name'] for comb in prediction_combinations]
column_names = [f"{shot}-{seed}" for seed in seed_numbers]
final_df = pd.DataFrame(index=row_names, columns=column_names)

# Function to process each combination
def process_combination(df_test, shot, seed, prediction_combinations):
    for comb in prediction_combinations:
        comb_name = comb['name']
        pred_comb = [df_test[col].tolist() for col in comb['lists']]

        all_predictions = np.array(pred_comb)
        all_pred_majority = np.mean(all_predictions, axis=0).round().astype(int)
        all_pred_majority = np.where(all_pred_majority == 0.5, random.choice(items), all_pred_majority)

        # Calculate classification report
        report = classification_report(y_test, all_pred_majority, target_names=class_names, output_dict=True , zero_division=0)
        macro_f1 = report['macro avg']['f1-score']

        # Update final_df with the macro F1 score
        final_df.at[comb_name, f"{shot}-{seed}"] = macro_f1

# Iterate over seed numbers and process each combination
for shot in shot_numbers:
    for seed in seed_numbers:
        # Read the CSV file for the current shot and seed to get y_test
        df_test = pd.read_csv(f"/Majority/feature_preds_{target_pair}_{shot}_{seed}.csv")
        y_test = df_test['y_test'].tolist()
        process_combination(df_test, shot, seed, prediction_combinations)

# Print the final DataFrame
print("Final DataFrame:")
print(final_df)

final_df.info()


# Convert columns to numeric data type
final_df = final_df.apply(pd.to_numeric, errors='ignore')

# Round the numeric columns to two decimal places
final_df = final_df.round(2)
final_df


# Calculate the average along each row and store it in a new column
final_df[f'{shot}-avg'] = final_df.mean(axis=1)
final_df = final_df.round(2)

shot_avg_column = f"{shot}-avg"
final_df = final_df[shot_avg_column]  # Selecting only the specified column

# Print the updated final DataFrame
print("Updated Final DataFrame:")
print(final_df)



path='/data/.../'
if shot==100:
    final_df.to_csv(path+ f'final_results_{target_pair}.csv')
    print(final_df)
# Read the existing CSV file into a DataFrame
else:
    existing_df = pd.read_csv(path+f'final_results_{target_pair}.csv', index_col=0)
    df = pd.concat([existing_df, final_df], axis=1)
    # Save the final DataFrame to a CSV file
    df.to_csv(path+ f'final_results_{target_pair}.csv')
    print(df)
