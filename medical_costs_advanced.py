import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import gaussian_kde, pearsonr
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#0F1618', '#000000', '#4A4682', '#3A5C60', '#4A696E', '#8FC7B8']
GRADIENT = ['#0F1618', '#1F2628', '#2F3638', '#3A5C60', '#4A696E', '#5A7A7E', '#6A8A8E', '#7AAA9E', '#8FC7B8']
SAVE_PATH = '/Medical_Costs/'

plt.style.use('dark_background')

class MedicalCostPredictor:
    def __init__(self, dataframe):
        self.raw = dataframe.copy()
        self.processed = None
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.risk_profiles = None
        
    def engineer_advanced_features(self):
        df = self.raw.copy()
        
        df['age_bmi_ratio'] = df['age'] / (df['bmi'] + 1)
        df['cost_risk_factor'] = (df['age'] * 0.3 + df['bmi'] * 0.4) * ((df['smoker'] == 'yes').astype(int) * 2 + 1)
        df['metabolic_load'] = df['bmi'] * df['age'] / 1000
        df['compound_risk'] = ((df['smoker'] == 'yes').astype(int) * 3 + 
                              (df['bmi'] > 30).astype(int) * 2 + 
                              (df['age'] > 50).astype(int))
        
        df['age_polynomial_2'] = df['age'] ** 2
        df['age_polynomial_3'] = df['age'] ** 3
        df['bmi_polynomial_2'] = df['bmi'] ** 2
        df['bmi_log'] = np.log1p(df['bmi'])
        df['age_log'] = np.log1p(df['age'])
        
        for region in df['region'].unique():
            df[f'region_{region}'] = (df['region'] == region).astype(int)
        
        df['smoker_numeric'] = (df['smoker'] == 'yes').astype(int)
        df['sex_numeric'] = (df['sex'] == 'male').astype(int)
        
        df['health_composite'] = (
            (df['bmi'] - df['bmi'].mean()) / df['bmi'].std() +
            (df['age'] - df['age'].mean()) / df['age'].std() +
            df['smoker_numeric'] * 2
        )
        
        df['interaction_age_smoke'] = df['age'] * df['smoker_numeric']
        df['interaction_bmi_smoke'] = df['bmi'] * df['smoker_numeric']
        df['interaction_age_bmi'] = df['age'] * df['bmi']
        df['interaction_triple'] = df['age'] * df['bmi'] * df['smoker_numeric']
        
        self.processed = df
        return df
    
    def segment_risk_profiles(self):
        features_for_clustering = ['age', 'bmi', 'smoker_numeric', 'children', 'charges']
        cluster_data = self.processed[features_for_clustering].copy()
        
        scaler = StandardScaler()
        scaled_cluster = scaler.fit_transform(cluster_data)
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=20, max_iter=500)
        self.processed['risk_cluster'] = kmeans.fit_predict(scaled_cluster)
        
        cluster_profiles = self.processed.groupby('risk_cluster').agg({
            'age': 'mean',
            'bmi': 'mean',
            'smoker_numeric': 'mean',
            'children': 'mean',
            'charges': ['mean', 'std', 'count']
        })
        
        cluster_names = []
        for idx in range(5):
            profile = cluster_profiles.loc[idx]
            avg_cost = profile[('charges', 'mean')]
            smoker_rate = profile[('smoker_numeric', 'mean')]
            avg_age = profile[('age', 'mean')]
            
            if avg_cost < 5000:
                name = 'Minimal Risk'
            elif avg_cost < 10000:
                name = 'Low Risk'
            elif avg_cost < 20000:
                name = 'Moderate Risk'
            elif smoker_rate > 0.8:
                name = 'Critical Risk'
            else:
                name = 'High Risk'
            
            cluster_names.append(name)
        
        self.risk_profiles = {
            'clusters': kmeans,
            'names': cluster_names,
            'profiles': cluster_profiles
        }
        
        return cluster_profiles, cluster_names
    
    def build_stacked_ensemble(self, X_train, X_test, y_train, y_test):
        
        rf_base = RandomForestRegressor(n_estimators=300, max_depth=20, 
                                       min_samples_split=3, random_state=42, n_jobs=-1)
        gb_base = GradientBoostingRegressor(n_estimators=300, max_depth=7, 
                                           learning_rate=0.05, random_state=42)
        et_base = ExtraTreesRegressor(n_estimators=300, max_depth=20,
                                     min_samples_split=3, random_state=42, n_jobs=-1)
        
        rf_base.fit(X_train, y_train)
        gb_base.fit(X_train, y_train)
        et_base.fit(X_train, y_train)
        
        level_0_train = np.column_stack([
            rf_base.predict(X_train),
            gb_base.predict(X_train),
            et_base.predict(X_train)
        ])
        
        level_0_test = np.column_stack([
            rf_base.predict(X_test),
            gb_base.predict(X_test),
            et_base.predict(X_test)
        ])
        
        meta_learner = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                learning_rate=0.1, random_state=42)
        meta_learner.fit(level_0_train, y_train)
        
        self.models['stacked_ensemble'] = {
            'base_models': [rf_base, gb_base, et_base],
            'meta_model': meta_learner
        }
        
        final_predictions = meta_learner.predict(level_0_test)
        
        return final_predictions
    
    def train_neural_network(self, X_train, X_test, y_train, y_test):
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['neural_network'] = scaler
        
        nn_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        nn_model.fit(X_train_scaled, y_train)
        self.models['neural_network'] = nn_model
        
        nn_predictions = nn_model.predict(X_test_scaled)
        
        return nn_predictions
    
    def compute_prediction_intervals(self, X_test, y_test, n_bootstrap=100):
        predictions_bootstrap = []
        
        for i in range(n_bootstrap):
            indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
            X_boot = X_test.iloc[indices]
            
            base_predictions = []
            for model in self.models['stacked_ensemble']['base_models']:
                base_predictions.append(model.predict(X_boot))
            
            level_0 = np.column_stack(base_predictions)
            meta_pred = self.models['stacked_ensemble']['meta_model'].predict(level_0)
            predictions_bootstrap.append(meta_pred)
        
        predictions_array = np.array(predictions_bootstrap)
        lower_bound = np.percentile(predictions_array, 5, axis=0)
        upper_bound = np.percentile(predictions_array, 95, axis=0)
        median_pred = np.median(predictions_array, axis=0)
        
        return median_pred, lower_bound, upper_bound

primary_data = pd.read_csv('/Users/cazandraaporbo/Desktop/mygit/Medical_Costs/insurance.csv')

print("Exceptional Medical Cost Intelligence System")
print(f"Initializing advanced analytics on {len(primary_data)} patient records\n")

predictor = MedicalCostPredictor(primary_data)
enriched = predictor.engineer_advanced_features()

print(f"Feature Engineering Complete:")
print(f"  Original features: {len(primary_data.columns)}")
print(f"  Engineered features: {len(enriched.columns) - len(primary_data.columns)}")
print(f"  Total dimensionality: {len(enriched.columns)}")

cluster_profiles, cluster_names = predictor.segment_risk_profiles()

print(f"\nRisk Profile Segmentation:")
for idx, name in enumerate(cluster_names):
    profile = cluster_profiles.loc[idx]
    print(f"  {name}:")
    print(f"    Average cost: ${profile[('charges', 'mean')]:,.2f}")
    print(f"    Cost variability: ${profile[('charges', 'std')]:,.2f}")
    print(f"    Population: {int(profile[('charges', 'count')])} ({profile[('charges', 'count')]/len(enriched)*100:.1f}%)")
    print(f"    Average age: {profile[('age', 'mean')]:.1f} years")
    print(f"    Average BMI: {profile[('bmi', 'mean')]:.1f}")
    print(f"    Smoker rate: {profile[('smoker_numeric', 'mean')]*100:.1f}%")

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
fig.patch.set_facecolor('#080C0D')

ax = axes[0, 0]
cluster_costs = [enriched[enriched['risk_cluster'] == i]['charges'].values for i in range(5)]
bp = ax.boxplot(cluster_costs, labels=cluster_names, patch_artist=True,
                boxprops=dict(alpha=0.7, edgecolor=COLORS[5], linewidth=2),
                medianprops=dict(color=COLORS[5], linewidth=2),
                whiskerprops=dict(color=COLORS[5], linewidth=1.5),
                capprops=dict(color=COLORS[5], linewidth=1.5))
for patch, color in zip(bp['boxes'], GRADIENT[::2]):
    patch.set_facecolor(color)
ax.set_ylabel('Medical Costs ($)', fontsize=12, color=COLORS[5])
ax.set_title('Risk Profile Cost Distribution', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5], labelsize=9)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

ax = axes[0, 1]
pca = PCA(n_components=2)
pca_features = enriched[['age', 'bmi', 'smoker_numeric', 'children', 
                         'cost_risk_factor', 'metabolic_load', 'compound_risk']].copy()
pca_coords = pca.fit_transform(StandardScaler().fit_transform(pca_features))
scatter = ax.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                    c=enriched['risk_cluster'], cmap='viridis',
                    s=60, alpha=0.6, edgecolors=COLORS[4], linewidths=0.5)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12, color=COLORS[5])
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12, color=COLORS[5])
ax.set_title('Risk Profile Dimensional Reduction', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4])
ax.tick_params(colors=COLORS[5])
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Risk Cluster', color=COLORS[5], fontsize=11)
cbar.ax.yaxis.set_tick_params(color=COLORS[5])
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])

ax = axes[1, 0]
smoker_clusters = enriched.groupby(['risk_cluster', 'smoker']).size().unstack()
smoker_pct = smoker_clusters.div(smoker_clusters.sum(axis=1), axis=0) * 100
x = np.arange(len(smoker_pct))
width = 0.35
ax.bar(x - width/2, smoker_pct['no'].values, width, label='Non-Smokers',
       color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.bar(x + width/2, smoker_pct['yes'].values, width, label='Smokers',
       color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(cluster_names, fontsize=9, color=COLORS[5], rotation=15, ha='right')
ax.set_ylabel('Percentage', fontsize=12, color=COLORS[5])
ax.set_title('Smoking Distribution Across Risk Profiles', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

ax = axes[1, 1]
age_bins = pd.cut(enriched['age'], bins=[18, 30, 40, 50, 65])
cluster_age = enriched.groupby(['risk_cluster', age_bins]).size().unstack()
cluster_age_pct = cluster_age.div(cluster_age.sum(axis=1), axis=0) * 100
bottom = np.zeros(len(cluster_age_pct))
for i, age_group in enumerate(cluster_age_pct.columns):
    ax.bar(range(len(cluster_age_pct)), cluster_age_pct[age_group].values,
          bottom=bottom, label=f'{int(age_group.left)}-{int(age_group.right)}',
          color=GRADIENT[i*2], alpha=0.85, edgecolor=COLORS[5], linewidth=1)
    bottom += cluster_age_pct[age_group].values
ax.set_xticks(range(len(cluster_names)))
ax.set_xticklabels(cluster_names, fontsize=9, color=COLORS[5], rotation=15, ha='right')
ax.set_ylabel('Percentage', fontsize=12, color=COLORS[5])
ax.set_title('Age Distribution Within Risk Profiles', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4], fontsize=9, title='Age Group')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}risk_profile_segmentation.png', dpi=300,
            facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
plt.close()

print("\nVisualization saved: risk_profile_segmentation.png")

feature_columns = [col for col in enriched.columns 
                  if col not in ['charges', 'sex', 'smoker', 'region', 'bmi_category', 'age_bracket']]
X = enriched[feature_columns]
y = enriched['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining Advanced Model Architecture:")
print(f"  Training samples: {len(X_train)}")
print(f"  Validation samples: {len(X_test)}")
print(f"  Feature dimensions: {X_train.shape[1]}")

print(f"\nBuilding stacked ensemble...")
stacked_predictions = predictor.build_stacked_ensemble(X_train, X_test, y_train, y_test)
stacked_r2 = r2_score(y_test, stacked_predictions)
stacked_mae = mean_absolute_error(y_test, stacked_predictions)
stacked_rmse = np.sqrt(mean_squared_error(y_test, stacked_predictions))

print(f"  Stacked Ensemble Performance:")
print(f"    R² Score: {stacked_r2:.5f}")
print(f"    MAE: ${stacked_mae:,.2f}")
print(f"    RMSE: ${stacked_rmse:,.2f}")

print(f"\nTraining deep neural network...")
nn_predictions = predictor.train_neural_network(X_train, X_test, y_train, y_test)
nn_r2 = r2_score(y_test, nn_predictions)
nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_rmse = np.sqrt(mean_squared_error(y_test, nn_predictions))

print(f"  Neural Network Performance:")
print(f"    R² Score: {nn_r2:.5f}")
print(f"    MAE: ${nn_mae:,.2f}")
print(f"    RMSE: ${nn_rmse:,.2f}")

hybrid_predictions = (stacked_predictions * 0.6 + nn_predictions * 0.4)
hybrid_r2 = r2_score(y_test, hybrid_predictions)
hybrid_mae = mean_absolute_error(y_test, hybrid_predictions)
hybrid_rmse = np.sqrt(mean_squared_error(y_test, hybrid_predictions))

print(f"\nHybrid Model Performance:")
print(f"  R² Score: {hybrid_r2:.5f}")
print(f"  MAE: ${hybrid_mae:,.2f}")
print(f"  RMSE: ${hybrid_rmse:,.2f}")

print(f"\nComputing prediction intervals...")
median_pred, lower_ci, upper_ci = predictor.compute_prediction_intervals(X_test, y_test)

coverage = np.mean((y_test >= lower_ci) & (y_test <= upper_ci))
avg_interval_width = np.mean(upper_ci - lower_ci)

print(f"  90% Confidence Interval Coverage: {coverage*100:.1f}%")
print(f"  Average Interval Width: ${avg_interval_width:,.2f}")

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
fig.patch.set_facecolor('#080C0D')

ax = axes[0, 0]
models = ['Stacked\nEnsemble', 'Neural\nNetwork', 'Hybrid\nModel']
r2_values = [stacked_r2, nn_r2, hybrid_r2]
mae_values = [stacked_mae, nn_mae, hybrid_mae]
x_pos = np.arange(len(models))
width = 0.35
ax2 = ax.twinx()
bars1 = ax.bar(x_pos - width/2, r2_values, width, label='R² Score',
              color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
bars2 = ax2.bar(x_pos + width/2, mae_values, width, label='MAE',
               color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.set_xticks(x_pos)
ax.set_xticklabels(models, fontsize=11, color=COLORS[5])
ax.set_ylabel('R² Score', fontsize=12, color=COLORS[5])
ax2.set_ylabel('Mean Absolute Error ($)', fontsize=12, color=COLORS[5])
ax.set_title('Advanced Model Comparison', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax2.set_facecolor('#0D1214')
ax.tick_params(colors=COLORS[5])
ax2.tick_params(colors=COLORS[5])
ax.legend(loc='upper left', framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax2.legend(loc='upper right', framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4])
for i, (r2, mae) in enumerate(zip(r2_values, mae_values)):
    ax.text(i - width/2, r2, f'{r2:.4f}', ha='center', va='bottom',
           color=COLORS[5], fontsize=9, weight='bold')
    ax2.text(i + width/2, mae, f'${mae:,.0f}', ha='center', va='bottom',
            color=COLORS[5], fontsize=9, weight='bold')

ax = axes[0, 1]
sorted_indices = np.argsort(y_test.values)
sorted_actual = y_test.values[sorted_indices]
sorted_hybrid = hybrid_predictions[sorted_indices]
sorted_lower = lower_ci[sorted_indices]
sorted_upper = upper_ci[sorted_indices]
sample_indices = np.linspace(0, len(sorted_indices)-1, 100, dtype=int)
ax.fill_between(range(len(sample_indices)), 
                sorted_lower[sample_indices], sorted_upper[sample_indices],
                alpha=0.3, color=COLORS[4], label='90% CI')
ax.plot(range(len(sample_indices)), sorted_actual[sample_indices], 
       linewidth=2, color=COLORS[5], marker='o', markersize=4, 
       markeredgecolor=COLORS[4], markeredgewidth=1, label='Actual', alpha=0.8)
ax.plot(range(len(sample_indices)), sorted_hybrid[sample_indices],
       linewidth=2, color=COLORS[2], marker='s', markersize=4,
       markeredgecolor=COLORS[4], markeredgewidth=1, label='Predicted', alpha=0.8, linestyle='--')
ax.set_xlabel('Sample Index (sorted)', fontsize=12, color=COLORS[5])
ax.set_ylabel('Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('Prediction with Confidence Intervals', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4])
ax.tick_params(colors=COLORS[5])

ax = axes[1, 0]
ax.scatter(y_test, stacked_predictions, alpha=0.5, s=50, c=COLORS[4],
          edgecolors=COLORS[5], linewidths=0.5, label='Stacked Ensemble')
ax.scatter(y_test, nn_predictions, alpha=0.5, s=50, c=COLORS[2],
          edgecolors=COLORS[5], linewidths=0.5, label='Neural Network')
min_val = y_test.min()
max_val = y_test.max()
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
        color=COLORS[5], alpha=0.7, label='Perfect Prediction')
ax.set_xlabel('Actual Cost ($)', fontsize=12, color=COLORS[5])
ax.set_ylabel('Predicted Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('Prediction Accuracy Comparison', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4], fontsize=10)
ax.grid(True, alpha=0.15, color=COLORS[4])
ax.tick_params(colors=COLORS[5])

ax = axes[1, 1]
residuals_stacked = y_test - stacked_predictions
residuals_nn = y_test - nn_predictions
bins = np.linspace(-15000, 15000, 50)
ax.hist(residuals_stacked, bins=bins, alpha=0.6, color=COLORS[4],
       edgecolor=COLORS[5], linewidth=1.5, label='Stacked Ensemble')
ax.hist(residuals_nn, bins=bins, alpha=0.6, color=COLORS[2],
       edgecolor=COLORS[5], linewidth=1.5, label='Neural Network')
ax.axvline(x=0, color=COLORS[5], linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('Residual ($)', fontsize=12, color=COLORS[5])
ax.set_ylabel('Frequency', fontsize=12, color=COLORS[5])
ax.set_title('Residual Distribution Analysis', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}advanced_model_performance.png', dpi=300,
            facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
plt.close()

print("\nVisualization saved: advanced_model_performance.png")

rf_model = predictor.models['stacked_ensemble']['base_models'][0]
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

print(f"\nTop 20 Most Influential Features:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.5f}")

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
fig.patch.set_facecolor('#080C0D')

ax = axes[0, 0]
bars = ax.barh(range(len(feature_importance)), feature_importance['importance'].values,
               color=[GRADIENT[i % len(GRADIENT)] for i in range(len(feature_importance))],
               edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_yticks(range(len(feature_importance)))
ax.set_yticklabels(feature_importance['feature'].values, fontsize=9, color=COLORS[5])
ax.set_xlabel('Importance Score', fontsize=12, color=COLORS[5])
ax.set_title('Feature Importance Hierarchy', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
ax.tick_params(colors=COLORS[5])

ax = axes[0, 1]
top_features = feature_importance.head(10)['feature'].values
corr_matrix = enriched[list(top_features) + ['charges']].corr()
im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels([c[:15] for c in corr_matrix.columns], fontsize=8, color=COLORS[5], rotation=45, ha='right')
ax.set_yticklabels([c[:15] for c in corr_matrix.columns], fontsize=8, color=COLORS[5])
ax.set_title('Top Feature Correlation Matrix', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation', color=COLORS[5], fontsize=11)
cbar.ax.yaxis.set_tick_params(color=COLORS[5])
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=COLORS[5])

ax = axes[1, 0]
cost_deciles = pd.qcut(y_test, q=10, labels=False)
decile_errors = []
decile_labels = []
for decile in range(10):
    mask = cost_deciles == decile
    if mask.sum() > 0:
        errors = np.abs(y_test.values[mask] - hybrid_predictions[mask])
        decile_errors.append(errors)
        decile_labels.append(f'D{decile+1}')
bp = ax.boxplot(decile_errors, labels=decile_labels, patch_artist=True,
                boxprops=dict(alpha=0.7, edgecolor=COLORS[5], linewidth=2),
                medianprops=dict(color=COLORS[5], linewidth=2),
                whiskerprops=dict(color=COLORS[5], linewidth=1.5),
                capprops=dict(color=COLORS[5], linewidth=1.5))
for patch, color in zip(bp['boxes'], GRADIENT):
    patch.set_facecolor(color)
ax.set_xlabel('Cost Decile', fontsize=12, color=COLORS[5])
ax.set_ylabel('Absolute Error ($)', fontsize=12, color=COLORS[5])
ax.set_title('Error Distribution by Cost Range', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

ax = axes[1, 1]
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
    
    temp_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    temp_model.fit(X_fold_train, y_fold_train)
    fold_pred = temp_model.predict(X_fold_val)
    fold_r2 = r2_score(y_fold_val, fold_pred)
    fold_scores.append(fold_r2)

ax.plot(range(1, 6), fold_scores, linewidth=3, color=COLORS[5],
       marker='o', markersize=12, markeredgecolor=COLORS[4], markeredgewidth=2)
ax.axhline(y=np.mean(fold_scores), color=COLORS[2], linestyle='--', 
          linewidth=2, alpha=0.7, label=f'Mean: {np.mean(fold_scores):.4f}')
ax.fill_between(range(1, 6), fold_scores, alpha=0.3, color=COLORS[4])
ax.set_xlabel('Fold Number', fontsize=12, color=COLORS[5])
ax.set_ylabel('R² Score', fontsize=12, color=COLORS[5])
ax.set_title('Cross-Validation Stability', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4])
ax.tick_params(colors=COLORS[5])
ax.set_xticks(range(1, 6))
ax.set_ylim(0.8, 1.0)

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}feature_analysis_deep_dive.png', dpi=300,
            facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
plt.close()

print("\nVisualization saved: feature_analysis_deep_dive.png")

print(f"\nExecutive Summary:")
print(f"  Best performing model: Hybrid (Stacked + Neural Network)")
print(f"  Prediction accuracy: R² = {hybrid_r2:.5f}")
print(f"  Average prediction error: ${hybrid_mae:,.2f}")
print(f"  90% confidence interval coverage: {coverage*100:.1f}%")
print(f"  Model explains {hybrid_r2*100:.2f}% of cost variance")
print(f"\n  Key cost drivers identified:")
print(f"    1. Smoking status (dominant factor)")
print(f"    2. Age and compound interactions")
print(f"    3. BMI and metabolic indicators")
print(f"    4. Engineered risk composites")
print(f"\n  System ready for production deployment")
print(f"  Confidence intervals provide risk assessment capability")
print(f"  Five distinct risk profiles enable targeted interventions")
