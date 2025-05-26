import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Load dataset ---
# Adjust the path if necessary
df = pd.read_csv('sandwich.csv')

# --- Preprocessing ---
df['butter_bin'] = df['butter'].map({'yes': 1, 'no': 0})

# --- Part (a): Descriptive analysis ---
print("=== Descriptive Statistics for antCount ===")
print(df['antCount'].describe(), "\n")

print("=== Category Counts ===")
print("Bread types:\n", df['bread'].value_counts(), "\n")
print("Toppings:\n", df['topping'].value_counts(), "\n")
print("Butter presence:\n", df['butter'].value_counts(), "\n")

bread_means = df.groupby('bread')['antCount'].mean()
topping_means = df.groupby('topping')['antCount'].mean()
butter_means = df.groupby('butter')['antCount'].mean()

print("=== Mean antCount by Bread ===\n", bread_means, "\n")
print("=== Mean antCount by Topping ===\n", topping_means, "\n")
print("=== Mean antCount by Butter ===\n", butter_means, "\n")


# Helper to save and show plots
def save_plot(title):
    filename = f"{title}.png"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.show()


# 1) Histogram
plt.figure()
df['antCount'].hist()
plt.xlabel('Ant Count')
plt.ylabel('Frequency')
save_plot('Distribution of Ant Counts')

# 2) Boxplot: antCount by bread type
plt.figure()
df.boxplot(column='antCount', by='bread')
plt.suptitle('')
plt.xlabel('Bread')
plt.ylabel('Ant Count')
save_plot('Ant Count by Bread Type')

# 3) Boxplot: antCount by topping
plt.figure()
df.boxplot(column='antCount', by='topping')
plt.suptitle('')
plt.xlabel('Topping')
plt.ylabel('Ant Count')
save_plot('Ant Count by Topping')

# 4) Bar plot: mean antCount by bread
plt.figure()
bread_means.plot(kind='bar')
plt.ylabel('Mean Ant Count')
save_plot('Mean Ant Count by Bread Type')

# 5) Bar plot: mean antCount by topping
plt.figure()
topping_means.plot(kind='bar')
plt.ylabel('Mean Ant Count')
save_plot('Mean Ant Count by Topping')

# 6) Bar plot: mean antCount by butter presence
plt.figure()
butter_means.plot(kind='bar')
plt.ylabel('Mean Ant Count')
save_plot('Mean Ant Count by Butter Presence')

# --- Part (b): Hypothesis tests ---
print("=== Hypothesis Tests ===\n")

# 1. Butter effect: two-sample t-test
yes_counts = df[df['butter'] == 'yes']['antCount']
no_counts = df[df['butter'] == 'no']['antCount']
t_butter, p_butter = stats.ttest_ind(yes_counts, no_counts)
print(f"T-test (butter yes vs no): t = {t_butter:.3f}, p = {p_butter:.3f}")

# 2. Bread type: one-way ANOVA
groups_bread = [df[df['bread'] == b]['antCount'] for b in df['bread'].unique()]
f_bread, p_bread = stats.f_oneway(*groups_bread)
print(f"ANOVA (bread types): F = {f_bread:.3f}, p = {p_bread:.3f}")

# 3. Topping: one-way ANOVA
groups_topping = [df[df['topping'] == t]['antCount'] for t in df['topping'].unique()]
f_topping, p_topping = stats.f_oneway(*groups_topping)
print(f"ANOVA (toppings): F = {f_topping:.3f}, p = {p_topping:.3f}")

# --- Random Forest model for variable importance ---
X = df[['bread', 'topping', 'butter_bin']]
y = df['antCount']

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first'), ['bread', 'topping'])
], remainder='passthrough')

rf_pipeline = Pipeline([
    ('pre', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=3, random_state=0))
])

rf_pipeline.fit(X, y)

# Extract feature importances
ohe = rf_pipeline.named_steps['pre'].named_transformers_['onehot']
feat_names = list(ohe.get_feature_names_out(['bread', 'topping'])) + ['butter_bin']
importances = rf_pipeline.named_steps['rf'].feature_importances_

# 7) Bar plot: feature importances
plt.figure()
plt.bar(feat_names, importances)
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
save_plot('Feature Importances (Random Forest)')
