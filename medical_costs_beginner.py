import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#0F1618', '#000000', '#4A4682', '#3A5C60', '#4A696E', '#8FC7B8']
GRADIENT = ['#0F1618', '#1F2628', '#2F3638', '#3A5C60', '#4A696E', '#5A7A7E', '#6A8A8E', '#7AAA9E', '#8FC7B8']
SAVE_PATH = '/Medical_Costs/'

plt.style.use('dark_background')

data = pd.read_csv('Medical_Costs/insurance.csv')

print("Medical Insurance Cost Analysis")
print(f"\nDataset contains {len(data)} individuals")
print(f"Average medical cost: ${data['charges'].mean():,.2f}")
print(f"Cost range: ${data['charges'].min():,.2f} to ${data['charges'].max():,.2f}")
print(f"\nAge range: {data['age'].min()} to {data['age'].max()} years")
print(f"Smokers: {(data['smoker'] == 'yes').sum()} ({(data['smoker'] == 'yes').sum()/len(data)*100:.1f}%)")
print(f"Average BMI: {data['bmi'].mean():.1f}")

fig, axes = plt.subplots(2, 3, figsize=(24, 14))
fig.patch.set_facecolor('#080C0D')

ax = axes[0, 0]
age_bins = pd.cut(data['age'], bins=[18, 30, 40, 50, 65], labels=['18-30', '30-40', '40-50', '50-65'])
age_costs = data.groupby(age_bins)['charges'].mean()
bars = ax.bar(range(len(age_costs)), age_costs.values, color=GRADIENT[1::2],
              edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_xticks(range(len(age_costs)))
ax.set_xticklabels(age_costs.index, fontsize=11, color=COLORS[5])
ax.set_ylabel('Average Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('Medical Costs by Age Group', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])
for i, (bar, val) in enumerate(zip(bars, age_costs.values)):
    ax.text(i, val, f'${val:,.0f}', ha='center', va='bottom', color=COLORS[5], fontsize=10, weight='bold')

print(f"\nAge Group Analysis:")
for group, cost in age_costs.items():
    print(f"  {group} years: ${cost:,.2f}")

ax = axes[0, 1]
smoker_costs = data.groupby('smoker')['charges'].mean()
colors_smoke = [COLORS[4], COLORS[2]]
bars = ax.bar(range(len(smoker_costs)), smoker_costs.values, color=colors_smoke,
              edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_xticks(range(len(smoker_costs)))
ax.set_xticklabels(['Non-Smoker', 'Smoker'], fontsize=12, color=COLORS[5])
ax.set_ylabel('Average Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('Smoking Impact on Medical Costs', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])
for i, (bar, val) in enumerate(zip(bars, smoker_costs.values)):
    ax.text(i, val, f'${val:,.0f}', ha='center', va='bottom', color=COLORS[5], fontsize=11, weight='bold')

print(f"\nSmoking Status Impact:")
print(f"  Non-smokers: ${smoker_costs['no']:,.2f}")
print(f"  Smokers: ${smoker_costs['yes']:,.2f}")
print(f"  Difference: ${smoker_costs['yes'] - smoker_costs['no']:,.2f} ({(smoker_costs['yes']/smoker_costs['no']-1)*100:.1f}% higher)")

ax = axes[0, 2]
bmi_categories = pd.cut(data['bmi'], bins=[0, 18.5, 25, 30, 100], 
                        labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
bmi_costs = data.groupby(bmi_categories)['charges'].mean()
bars = ax.bar(range(len(bmi_costs)), bmi_costs.values, color=GRADIENT[::2],
              edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_xticks(range(len(bmi_costs)))
ax.set_xticklabels(bmi_costs.index, fontsize=11, color=COLORS[5], rotation=15)
ax.set_ylabel('Average Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('BMI Category vs Medical Costs', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])
for i, (bar, val) in enumerate(zip(bars, bmi_costs.values)):
    ax.text(i, val, f'${val:,.0f}', ha='center', va='bottom', color=COLORS[5], fontsize=10, weight='bold')

print(f"\nBMI Category Analysis:")
for category, cost in bmi_costs.items():
    count = (bmi_categories == category).sum()
    print(f"  {category}: ${cost:,.2f} (n={count})")

ax = axes[1, 0]
region_costs = data.groupby('region')['charges'].mean().sort_values(ascending=True)
bars = ax.barh(range(len(region_costs)), region_costs.values,
               color=[GRADIENT[i*2] for i in range(len(region_costs))],
               edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_yticks(range(len(region_costs)))
ax.set_yticklabels([r.title() for r in region_costs.index], fontsize=11, color=COLORS[5])
ax.set_xlabel('Average Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('Regional Cost Differences', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
ax.tick_params(colors=COLORS[5])
for i, (bar, val) in enumerate(zip(bars, region_costs.values)):
    ax.text(val, i, f'  ${val:,.0f}', va='center', ha='left', color=COLORS[5], fontsize=10, weight='bold')

print(f"\nRegional Variation:")
for region, cost in region_costs.items():
    print(f"  {region.title()}: ${cost:,.2f}")

ax = axes[1, 1]
children_costs = data.groupby('children')['charges'].mean()
ax.plot(children_costs.index, children_costs.values, linewidth=3, color=COLORS[5],
        marker='o', markersize=10, markeredgecolor=COLORS[4], markeredgewidth=2)
ax.fill_between(children_costs.index, children_costs.values, alpha=0.3, color=COLORS[4])
ax.set_xlabel('Number of Children', fontsize=12, color=COLORS[5])
ax.set_ylabel('Average Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('Family Size Impact', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4])
ax.tick_params(colors=COLORS[5])

print(f"\nFamily Size Analysis:")
for kids, cost in children_costs.items():
    count = (data['children'] == kids).sum()
    print(f"  {kids} children: ${cost:,.2f} (n={count})")

ax = axes[1, 2]
scatter = ax.scatter(data['age'], data['charges'], c=data['bmi'], cmap='viridis',
                    s=50, alpha=0.6, edgecolors=COLORS[4], linewidths=0.5)
ax.set_xlabel('Age (years)', fontsize=12, color=COLORS[5])
ax.set_ylabel('Medical Costs ($)', fontsize=12, color=COLORS[5])
ax.set_title('Age vs Cost (colored by BMI)', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4])
ax.tick_params(colors=COLORS[5])
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('BMI', color=COLORS[5], fontsize=11)
cbar.ax.yaxis.set_tick_params(color=COLORS[5])
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}basic_cost_exploration.png', dpi=300, 
            facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
plt.close()

print("\nVisualization saved: basic_cost_exploration.png")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor('#080C0D')

ax = axes[0, 0]
numeric_cols = ['age', 'bmi', 'children', 'charges']
correlation_matrix = data[numeric_cols].corr()
im = ax.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels([c.upper() for c in numeric_cols], fontsize=11, color=COLORS[5])
ax.set_yticklabels([c.upper() for c in numeric_cols], fontsize=11, color=COLORS[5])
ax.set_title('Feature Correlations', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        val = correlation_matrix.iloc[i, j]
        text_color = COLORS[0] if abs(val) > 0.5 else COLORS[5]
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
               color=text_color, fontsize=11, weight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation', color=COLORS[5], fontsize=11)
cbar.ax.yaxis.set_tick_params(color=COLORS[5])
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])

print(f"\nCorrelation with Medical Costs:")
cost_corr = correlation_matrix['charges'].drop('charges').sort_values(ascending=False)
for feature, corr in cost_corr.items():
    print(f"  {feature.upper()}: {corr:.3f}")

ax = axes[0, 1]
smoker_age_cost = data.groupby(['smoker', pd.cut(data['age'], bins=5)])['charges'].mean().unstack()
x = range(len(smoker_age_cost.columns))
width = 0.35
ax.bar([i - width/2 for i in x], smoker_age_cost.loc['no'].values, width,
       label='Non-Smoker', color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.bar([i + width/2 for i in x], smoker_age_cost.loc['yes'].values, width,
       label='Smoker', color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.set_xticks(x)
labels = [f"{int(cat.left)}-{int(cat.right)}" for cat in smoker_age_cost.columns]
ax.set_xticklabels(labels, fontsize=10, color=COLORS[5])
ax.set_ylabel('Average Cost ($)', fontsize=12, color=COLORS[5])
ax.set_xlabel('Age Range', fontsize=12, color=COLORS[5])
ax.set_title('Smoking Effect Across Age Groups', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

ax = axes[1, 0]
cost_distribution = [
    data[data['charges'] < 5000]['charges'],
    data[(data['charges'] >= 5000) & (data['charges'] < 15000)]['charges'],
    data[(data['charges'] >= 15000) & (data['charges'] < 30000)]['charges'],
    data[data['charges'] >= 30000]['charges']
]
labels_dist = ['< $5K', '$5K-$15K', '$15K-$30K', '> $30K']
bp = ax.boxplot(cost_distribution, labels=labels_dist, patch_artist=True,
                boxprops=dict(facecolor=COLORS[4], alpha=0.7, edgecolor=COLORS[5], linewidth=2),
                medianprops=dict(color=COLORS[5], linewidth=2),
                whiskerprops=dict(color=COLORS[5], linewidth=1.5),
                capprops=dict(color=COLORS[5], linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor=COLORS[2], markersize=5,
                               markeredgecolor=COLORS[5], alpha=0.6))
ax.set_ylabel('Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('Cost Distribution by Range', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

print(f"\nCost Distribution:")
print(f"  < $5,000: {len(cost_distribution[0])} people ({len(cost_distribution[0])/len(data)*100:.1f}%)")
print(f"  $5,000-$15,000: {len(cost_distribution[1])} people ({len(cost_distribution[1])/len(data)*100:.1f}%)")
print(f"  $15,000-$30,000: {len(cost_distribution[2])} people ({len(cost_distribution[2])/len(data)*100:.1f}%)")
print(f"  > $30,000: {len(cost_distribution[3])} people ({len(cost_distribution[3])/len(data)*100:.1f}%)")

ax = axes[1, 1]
sex_smoker = data.groupby(['sex', 'smoker'])['charges'].mean().unstack()
x = np.arange(len(sex_smoker.index))
width = 0.35
ax.bar(x - width/2, sex_smoker['no'].values, width, label='Non-Smoker',
       color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.bar(x + width/2, sex_smoker['yes'].values, width, label='Smoker',
       color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(['Female', 'Male'], fontsize=12, color=COLORS[5])
ax.set_ylabel('Average Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('Gender and Smoking Combined Effect', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

print(f"\nGender and Smoking Interaction:")
print(f"  Female non-smokers: ${sex_smoker.loc['female', 'no']:,.2f}")
print(f"  Female smokers: ${sex_smoker.loc['female', 'yes']:,.2f}")
print(f"  Male non-smokers: ${sex_smoker.loc['male', 'no']:,.2f}")
print(f"  Male smokers: ${sex_smoker.loc['male', 'yes']:,.2f}")

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}statistical_patterns.png', dpi=300,
            facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
plt.close()

print("\nVisualization saved: statistical_patterns.png")

data_encoded = data.copy()
data_encoded['sex'] = (data_encoded['sex'] == 'male').astype(int)
data_encoded['smoker'] = (data_encoded['smoker'] == 'yes').astype(int)
region_dummies = pd.get_dummies(data_encoded['region'], prefix='region', drop_first=True)
data_encoded = pd.concat([data_encoded, region_dummies], axis=1)

predictors = ['age', 'sex', 'bmi', 'children', 'smoker'] + list(region_dummies.columns)
X = data_encoded[predictors]
y = data_encoded['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nLinear Regression Model Performance:")
print(f"  Training R²: {r2_train:.3f}")
print(f"  Testing R²: {r2_test:.3f}")
print(f"  Mean Absolute Error: ${mae_test:,.2f}")
print(f"  Root Mean Squared Error: ${rmse_test:,.2f}")

feature_importance = pd.DataFrame({
    'feature': predictors,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(f"\nFeature Importance (Linear Model Coefficients):")
for idx, row in feature_importance.iterrows():
    sign = '+' if row['coefficient'] > 0 else ''
    print(f"  {row['feature']}: {sign}${row['coefficient']:,.2f}")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.patch.set_facecolor('#080C0D')

ax = axes[0, 0]
bars = ax.barh(range(len(feature_importance)), feature_importance['coefficient'].abs().values,
               color=[GRADIENT[i % len(GRADIENT)] for i in range(len(feature_importance))],
               edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['feature'].values, fontsize=11, color=COLORS[5])
ax.set_xlabel('Coefficient Magnitude', fontsize=12, color=COLORS[5])
ax.set_title('Feature Impact on Medical Costs', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
ax.tick_params(colors=COLORS[5])

ax = axes[0, 1]
ax.scatter(y_test, y_pred_test, alpha=0.6, s=50, c=COLORS[4], edgecolors=COLORS[5], linewidths=0.5)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, color=COLORS[2], alpha=0.7)
ax.set_xlabel('Actual Cost ($)', fontsize=12, color=COLORS[5])
ax.set_ylabel('Predicted Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('Prediction Accuracy', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4])
ax.tick_params(colors=COLORS[5])
ax.text(0.05, 0.95, f'R² = {r2_test:.3f}', transform=ax.transAxes,
        fontsize=12, color=COLORS[5], weight='bold', va='top',
        bbox=dict(boxstyle='round', facecolor='#080C0D', edgecolor=COLORS[4], alpha=0.8))

ax = axes[1, 0]
residuals = y_test - y_pred_test
ax.scatter(y_pred_test, residuals, alpha=0.6, s=50, c=COLORS[4], edgecolors=COLORS[5], linewidths=0.5)
ax.axhline(y=0, color=COLORS[2], linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Predicted Cost ($)', fontsize=12, color=COLORS[5])
ax.set_ylabel('Residual ($)', fontsize=12, color=COLORS[5])
ax.set_title('Residual Analysis', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4])
ax.tick_params(colors=COLORS[5])

ax = axes[1, 1]
sorted_residuals = np.sort(residuals)
theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
ax.scatter(theoretical_quantiles, sorted_residuals, alpha=0.6, s=50, 
          c=COLORS[4], edgecolors=COLORS[5], linewidths=0.5)
ax.plot([theoretical_quantiles.min(), theoretical_quantiles.max()],
        [sorted_residuals.min(), sorted_residuals.max()],
        'r--', linewidth=2, color=COLORS[2], alpha=0.7)
ax.set_xlabel('Theoretical Quantiles', fontsize=12, color=COLORS[5])
ax.set_ylabel('Sample Quantiles', fontsize=12, color=COLORS[5])
ax.set_title('Q-Q Plot for Residuals', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4])
ax.tick_params(colors=COLORS[5])

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}prediction_model_results.png', dpi=300,
            facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
plt.close()

print("\nVisualization saved: prediction_model_results.png")

print(f"\nKey Insights:")
print(f"  1. Smoking status has the largest impact on medical costs")
print(f"  2. BMI shows moderate positive correlation with costs")
print(f"  3. Age correlates with higher costs, especially for smokers")
print(f"  4. Regional differences exist but are less significant")
print(f"  5. Number of children has minimal impact on individual costs")
print(f"\nModel explains {r2_test*100:.1f}% of cost variation")
print(f"Average prediction error: ${mae_test:,.2f}")
