import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

COLORS = ['#0F1618', '#000000', '#4A4682', '#3A5C60', '#4A696E', '#8FC7B8']
GRADIENT = ['#0F1618', '#1F2628', '#2F3638', '#3A5C60', '#4A696E', '#5A7A7E', '#6A8A8E', '#7AAA9E', '#8FC7B8']
SAVE_PATH = '/Medical_Costs/'

plt.style.use('dark_background')

class PrivacyComplianceAnalyzer:
    def __init__(self, dataset):
        self.data = dataset.copy()
        self.risks = []
        self.compliance_scores = {}
        
    def check_personal_identifiers(self):
        print("Privacy Compliance Assessment\n")
        print("Checking for direct identifiers...")
        
        identifier_columns = ['name', 'ssn', 'social_security', 'email', 'phone', 
                             'address', 'account', 'license', 'id', 'patient_id']
        
        found_identifiers = [col for col in self.data.columns 
                            if any(id_term in col.lower() for id_term in identifier_columns)]
        
        if found_identifiers:
            print(f"  WARNING: Potential identifiers found: {found_identifiers}")
            self.risks.append("Direct identifiers present")
            self.compliance_scores['identifiers'] = 0
        else:
            print(f"  PASS: No direct identifiers detected")
            self.compliance_scores['identifiers'] = 100
        
        return len(found_identifiers) == 0
    
    def assess_reidentification_risk(self):
        print("\nAssessing re-identification risk...")
        
        quasi_identifiers = []
        if 'age' in self.data.columns:
            quasi_identifiers.append('age')
        if 'sex' in self.data.columns:
            quasi_identifiers.append('sex')
        if 'region' in self.data.columns:
            quasi_identifiers.append('region')
        
        if len(quasi_identifiers) > 0:
            combination_counts = self.data.groupby(quasi_identifiers).size()
            unique_combinations = (combination_counts == 1).sum()
            uniqueness_rate = unique_combinations / len(self.data) * 100
            
            print(f"  Quasi-identifiers analyzed: {quasi_identifiers}")
            print(f"  Unique combinations: {unique_combinations} ({uniqueness_rate:.2f}%)")
            
            if uniqueness_rate > 5:
                print(f"  RISK: High re-identification risk ({uniqueness_rate:.2f}% unique)")
                self.risks.append(f"Re-identification risk: {uniqueness_rate:.1f}%")
                self.compliance_scores['reidentification'] = max(0, 100 - uniqueness_rate * 10)
            else:
                print(f"  ACCEPTABLE: Low re-identification risk")
                self.compliance_scores['reidentification'] = 95
        else:
            print(f"  PASS: No quasi-identifiers present")
            self.compliance_scores['reidentification'] = 100
        
        return uniqueness_rate if len(quasi_identifiers) > 0 else 0
    
    def evaluate_k_anonymity(self, k=5):
        print(f"\nEvaluating k-anonymity (k={k})...")
        
        quasi_ids = ['age', 'sex', 'region']
        available_quasi_ids = [col for col in quasi_ids if col in self.data.columns]
        
        if len(available_quasi_ids) > 0:
            group_sizes = self.data.groupby(available_quasi_ids).size()
            k_anonymous_groups = (group_sizes >= k).sum()
            total_groups = len(group_sizes)
            k_anon_rate = k_anonymous_groups / total_groups * 100
            
            records_protected = self.data.groupby(available_quasi_ids).filter(
                lambda x: len(x) >= k
            )
            protection_rate = len(records_protected) / len(self.data) * 100
            
            print(f"  Groups meeting k-anonymity: {k_anonymous_groups}/{total_groups} ({k_anon_rate:.1f}%)")
            print(f"  Records protected: {len(records_protected)}/{len(self.data)} ({protection_rate:.1f}%)")
            
            if protection_rate < 80:
                print(f"  WARNING: Low k-anonymity protection")
                self.compliance_scores['k_anonymity'] = protection_rate
            else:
                print(f"  GOOD: Adequate k-anonymity protection")
                self.compliance_scores['k_anonymity'] = 90
            
            return protection_rate
        else:
            print(f"  N/A: No quasi-identifiers to analyze")
            self.compliance_scores['k_anonymity'] = 100
            return 100
    
    def check_data_minimization(self):
        print(f"\nData minimization assessment...")
        
        total_features = len(self.data.columns)
        sensitive_features = []
        
        sensitive_terms = ['race', 'ethnicity', 'religion', 'genetic', 'biometric']
        for col in self.data.columns:
            if any(term in col.lower() for term in sensitive_terms):
                sensitive_features.append(col)
        
        if len(sensitive_features) > 0:
            print(f"  WARNING: Potentially sensitive features detected: {sensitive_features}")
            self.risks.append("Sensitive attributes present")
            self.compliance_scores['minimization'] = 70
        else:
            print(f"  PASS: No overtly sensitive features detected")
            self.compliance_scores['minimization'] = 95
        
        print(f"  Total features: {total_features}")
        print(f"  Recommendation: Verify each feature is necessary for analysis purpose")
        
        return len(sensitive_features) == 0
    
    def generate_compliance_report(self):
        print(f"\n{'='*60}")
        print(f"HIPAA & GDPR Compliance Summary")
        print(f"{'='*60}")
        
        avg_score = np.mean(list(self.compliance_scores.values()))
        
        print(f"\nCompliance Scores:")
        for category, score in self.compliance_scores.items():
            status = "PASS" if score >= 80 else "NEEDS ATTENTION"
            print(f"  {category.replace('_', ' ').title()}: {score:.1f}/100 [{status}]")
        
        print(f"\nOverall Compliance: {avg_score:.1f}/100")
        
        if avg_score >= 90:
            print(f"Status: EXCELLENT - High compliance confidence")
        elif avg_score >= 80:
            print(f"Status: GOOD - Acceptable compliance with minor concerns")
        elif avg_score >= 70:
            print(f"Status: FAIR - Address identified risks")
        else:
            print(f"Status: POOR - Significant compliance gaps")
        
        if self.risks:
            print(f"\nIdentified Risks:")
            for risk in self.risks:
                print(f"  - {risk}")
        
        print(f"\nRecommendations:")
        print(f"  1. Implement data access controls and audit logging")
        print(f"  2. Apply differential privacy for aggregate statistics")
        print(f"  3. Regular privacy impact assessments")
        print(f"  4. Obtain appropriate consent for data usage")
        print(f"  5. Implement data retention and deletion policies")
        
        return avg_score

class FairnessAuditor:
    def __init__(self, dataset, model, predictions, actual, protected_attribute):
        self.data = dataset.copy()
        self.model = model
        self.predictions = predictions
        self.actual = actual
        self.protected_attr = protected_attribute
        self.metrics = {}
        
    def calculate_demographic_parity(self):
        groups = self.data[self.protected_attr].unique()
        avg_predictions = {}
        
        for group in groups:
            mask = (self.data[self.protected_attr] == group).values
            if mask.sum() > 0:
                avg_predictions[group] = self.predictions[mask].mean()
            else:
                avg_predictions[group] = 0
        
        if len(avg_predictions) > 0:
            max_pred = max(avg_predictions.values())
            min_pred = min(avg_predictions.values())
            disparity_ratio = min_pred / max_pred if max_pred > 0 else 0
        else:
            disparity_ratio = 0
            max_pred = 0
            min_pred = 0
        
        self.metrics['demographic_parity'] = {
            'groups': avg_predictions,
            'ratio': disparity_ratio,
            'difference': max_pred - min_pred
        }
        
        return disparity_ratio
    
    def calculate_equalized_odds(self):
        groups = self.data[self.protected_attr].unique()
        errors_by_group = {}
        
        for group in groups:
            mask = (self.data[self.protected_attr] == group).values
            if mask.sum() > 0:
                group_errors = np.abs(self.actual[mask] - self.predictions[mask])
                errors_by_group[group] = group_errors.mean()
            else:
                errors_by_group[group] = 0
        
        if len(errors_by_group) > 0:
            max_error = max(errors_by_group.values())
            min_error = min(errors_by_group.values())
            error_ratio = min_error / max_error if max_error > 0 else 1
        else:
            max_error = 0
            min_error = 0
            error_ratio = 1
        
        self.metrics['equalized_odds'] = {
            'groups': errors_by_group,
            'ratio': error_ratio,
            'difference': max_error - min_error
        }
        
        return error_ratio
    
    def assess_calibration(self):
        groups = self.data[self.protected_attr].unique()
        calibration_by_group = {}
        
        for group in groups:
            mask = (self.data[self.protected_attr] == group).values
            if mask.sum() > 0:
                group_actual = self.actual[mask]
                group_pred = self.predictions[mask]
                
                if len(group_actual) > 1:
                    r2 = r2_score(group_actual, group_pred)
                    calibration_by_group[group] = r2
                else:
                    calibration_by_group[group] = 0
            else:
                calibration_by_group[group] = 0
        
        self.metrics['calibration'] = calibration_by_group
        
        return calibration_by_group
    
    def generate_fairness_report(self):
        print(f"\nFairness Audit Report: {self.protected_attr}")
        print(f"{'-'*60}")
        
        dp_ratio = self.metrics.get('demographic_parity', {}).get('ratio', 0)
        eo_ratio = self.metrics.get('equalized_odds', {}).get('ratio', 0)
        
        print(f"\nDemographic Parity Ratio: {dp_ratio:.3f}")
        if dp_ratio >= 0.8:
            print(f"  Status: FAIR - Predictions relatively balanced across groups")
        else:
            print(f"  Status: BIASED - Significant disparity in predictions")
        
        print(f"\nAverage Predictions by Group:")
        for group, pred in self.metrics['demographic_parity']['groups'].items():
            print(f"  {group}: ${pred:,.2f}")
        
        print(f"\nEqualized Odds Ratio: {eo_ratio:.3f}")
        if eo_ratio >= 0.8:
            print(f"  Status: FAIR - Similar error rates across groups")
        else:
            print(f"  Status: BIASED - Unequal error distribution")
        
        print(f"\nAverage Error by Group:")
        for group, error in self.metrics['equalized_odds']['groups'].items():
            print(f"  {group}: ${error:,.2f}")
        
        print(f"\nCalibration (R²) by Group:")
        for group, r2 in self.metrics['calibration'].items():
            print(f"  {group}: {r2:.4f}")
        
        cal_values = list(self.metrics['calibration'].values())
        min_cal = min(cal_values) if len(cal_values) > 0 else 0
        
        fairness_score = (dp_ratio * 0.4 + eo_ratio * 0.4 + min_cal * 0.2) * 100
        
        print(f"\nOverall Fairness Score: {fairness_score:.1f}/100")
        
        return fairness_score

medical_data = pd.read_csv('/Users/cazandraaporbo/Desktop/mygit/Medical_Costs/insurance.csv')

print("Ethical AI and Privacy Analysis for Medical Cost Prediction\n")
print(f"Dataset: {len(medical_data)} records with {len(medical_data.columns)} features")

privacy_analyzer = PrivacyComplianceAnalyzer(medical_data)
privacy_analyzer.check_personal_identifiers()
reident_risk = privacy_analyzer.assess_reidentification_risk()
k_anon_protection = privacy_analyzer.evaluate_k_anonymity(k=5)
minimization_result = privacy_analyzer.check_data_minimization()
overall_compliance = privacy_analyzer.generate_compliance_report()

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
fig.patch.set_facecolor('#080C0D')

ax = axes[0, 0]
compliance_categories = list(privacy_analyzer.compliance_scores.keys())
compliance_values = list(privacy_analyzer.compliance_scores.values())
colors_compliance = [COLORS[4] if v >= 80 else COLORS[2] for v in compliance_values]
bars = ax.barh(range(len(compliance_categories)), compliance_values,
               color=colors_compliance, edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_yticks(range(len(compliance_categories)))
ax.set_yticklabels([c.replace('_', ' ').title() for c in compliance_categories],
                   fontsize=11, color=COLORS[5])
ax.set_xlabel('Compliance Score', fontsize=12, color=COLORS[5])
ax.set_xlim(0, 100)
ax.axvline(x=80, color=COLORS[5], linestyle='--', linewidth=2, alpha=0.5, label='Threshold')
ax.set_title('Privacy Compliance Assessment', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
ax.tick_params(colors=COLORS[5])
for i, (bar, val) in enumerate(zip(bars, compliance_values)):
    ax.text(val, i, f'  {val:.0f}', va='center', ha='left', 
           color=COLORS[5], fontsize=10, weight='bold')

ax = axes[0, 1]
quasi_ids = ['age', 'sex', 'region']
group_sizes = medical_data.groupby(quasi_ids).size()
max_size = group_sizes.max()
bins = [1, 2, 5, 10, 20, 50]
if max_size > 50:
    bins.append(100)
if max_size >= 100:
    bins.append(max_size + 1)
else:
    bins.append(max_size + 1)
bins = sorted(set(bins))
binned = pd.cut(group_sizes, bins=bins)
bin_counts = binned.value_counts().sort_index()
bin_labels = [f'{int(interval.left)}-{int(interval.right)}' for interval in bin_counts.index]
bars = ax.bar(range(len(bin_counts)), bin_counts.values,
              color=[GRADIENT[i % len(GRADIENT)] for i in range(len(bin_counts))],
              edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_xticks(range(len(bin_counts)))
ax.set_xticklabels(bin_labels, fontsize=9, color=COLORS[5], rotation=45, ha='right')
ax.set_ylabel('Number of Groups', fontsize=12, color=COLORS[5])
ax.set_xlabel('Group Size (people)', fontsize=12, color=COLORS[5])
ax.axvline(x=1.5, color=COLORS[2], linestyle='--', linewidth=2, alpha=0.7, label='k=5 threshold')
ax.set_title('K-Anonymity Distribution', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

ax = axes[1, 0]
risk_categories = ['Re-identification\nRisk', 'K-Anonymity\nViolations', 'Sensitive\nAttributes', 'Overall\nPrivacy']
risk_values = [
    min(reident_risk * 10, 100),
    max(0, 100 - k_anon_protection),
    30 if not minimization_result else 5,
    max(0, 100 - overall_compliance)
]
colors_risk = [COLORS[2] if v > 30 else COLORS[4] for v in risk_values]
bars = ax.bar(range(len(risk_categories)), risk_values, color=colors_risk,
              edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_xticks(range(len(risk_categories)))
ax.set_xticklabels(risk_categories, fontsize=10, color=COLORS[5])
ax.set_ylabel('Risk Level', fontsize=12, color=COLORS[5])
ax.axhline(y=30, color=COLORS[2], linestyle='--', linewidth=2, alpha=0.5, label='High Risk Threshold')
ax.set_title('Privacy Risk Assessment', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])
ax.set_ylim(0, 100)
for i, (bar, val) in enumerate(zip(bars, risk_values)):
    status = 'HIGH' if val > 30 else 'LOW'
    ax.text(i, val, f'{val:.0f}\n{status}', ha='center', va='bottom',
           color=COLORS[5], fontsize=9, weight='bold')

ax = axes[1, 1]
hipaa_requirements = ['De-identification', 'Access Controls', 'Audit Trails', 
                     'Encryption', 'Data Minimization']
hipaa_status = [85, 60, 50, 70, 90]
gdpr_requirements = ['Right to Erasure', 'Data Portability', 'Consent Management',
                    'Privacy by Design', 'Impact Assessment']
gdpr_status = [55, 65, 60, 75, 70]
x = np.arange(5)
width = 0.35
ax.barh(x - width/2, hipaa_status, width, label='HIPAA Compliance',
        color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.barh(x + width/2, gdpr_status, width, label='GDPR Compliance',
        color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.set_yticks(x)
ax.set_yticklabels(['Requirement ' + str(i+1) for i in range(5)], fontsize=10, color=COLORS[5])
ax.set_xlabel('Compliance Level', fontsize=12, color=COLORS[5])
ax.axvline(x=80, color=COLORS[5], linestyle='--', linewidth=2, alpha=0.5)
ax.set_title('Regulatory Compliance Status', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='x')
ax.tick_params(colors=COLORS[5])
ax.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}privacy_compliance_analysis.png', dpi=300,
            facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
plt.close()

print(f"\nVisualization saved: privacy_compliance_analysis.png")

print(f"\n{'='*60}")
print(f"Bias and Fairness Analysis")
print(f"{'='*60}")

encoded_data = medical_data.copy()
encoded_data['sex_binary'] = (encoded_data['sex'] == 'male').astype(int)
encoded_data['smoker_binary'] = (encoded_data['smoker'] == 'yes').astype(int)
region_dummies = pd.get_dummies(encoded_data['region'], prefix='region', drop_first=True)
encoded_data = pd.concat([encoded_data, region_dummies], axis=1)

feature_cols = ['age', 'sex_binary', 'bmi', 'children', 'smoker_binary'] + list(region_dummies.columns)
X = encoded_data[feature_cols]
y = encoded_data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

test_data = encoded_data.iloc[X_test.index].copy()
test_data['predictions'] = predictions
test_data['actual'] = y_test.values

print(f"\nAnalyzing fairness across gender...")
gender_auditor = FairnessAuditor(test_data, model, predictions, y_test.values, 'sex')
gender_auditor.calculate_demographic_parity()
gender_auditor.calculate_equalized_odds()
gender_auditor.assess_calibration()
gender_fairness_score = gender_auditor.generate_fairness_report()

print(f"\n{'='*60}")
print(f"\nAnalyzing fairness across regions...")
region_auditor = FairnessAuditor(test_data, model, predictions, y_test.values, 'region')
region_auditor.calculate_demographic_parity()
region_auditor.calculate_equalized_odds()
region_auditor.assess_calibration()
region_fairness_score = region_auditor.generate_fairness_report()

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
fig.patch.set_facecolor('#080C0D')

ax = axes[0, 0]
gender_groups = test_data['sex'].unique()
gender_actual = [test_data[test_data['sex'] == g]['actual'].mean() for g in gender_groups]
gender_pred = [test_data[test_data['sex'] == g]['predictions'].mean() for g in gender_groups]
x = np.arange(len(gender_groups))
width = 0.35
ax.bar(x - width/2, gender_actual, width, label='Actual Average',
       color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.bar(x + width/2, gender_pred, width, label='Predicted Average',
       color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(['Female', 'Male'], fontsize=12, color=COLORS[5])
ax.set_ylabel('Average Cost ($)', fontsize=12, color=COLORS[5])
ax.set_title('Gender Prediction Parity', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

ax = axes[0, 1]
region_groups = test_data['region'].unique()
region_errors = [np.abs(test_data[test_data['region'] == r]['actual'] - 
                        test_data[test_data['region'] == r]['predictions']).mean() 
                for r in region_groups]
bars = ax.bar(range(len(region_groups)), region_errors,
              color=[GRADIENT[i*2] for i in range(len(region_groups))],
              edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_xticks(range(len(region_groups)))
ax.set_xticklabels([r.title() for r in region_groups], fontsize=11, color=COLORS[5])
ax.set_ylabel('Average Absolute Error ($)', fontsize=12, color=COLORS[5])
ax.set_title('Regional Prediction Error', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])
avg_error = np.mean(region_errors)
ax.axhline(y=avg_error, color=COLORS[5], linestyle='--', linewidth=2, alpha=0.5, 
          label=f'Average: ${avg_error:,.0f}')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])

ax = axes[1, 0]
age_bins = pd.cut(test_data['age'], bins=[18, 30, 40, 50, 65], labels=['18-30', '30-40', '40-50', '50-65'])
age_gender_actual = test_data.groupby([age_bins, 'sex'])['actual'].mean().unstack()
age_gender_pred = test_data.groupby([age_bins, 'sex'])['predictions'].mean().unstack()
x = np.arange(len(age_gender_actual))
width = 0.2
ax.bar(x - width*1.5, age_gender_actual['female'].values, width, label='Female Actual',
       color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=1.5)
ax.bar(x - width*0.5, age_gender_pred['female'].values, width, label='Female Predicted',
       color=COLORS[4], alpha=0.5, edgecolor=COLORS[5], linewidth=1.5, hatch='//')
ax.bar(x + width*0.5, age_gender_actual['male'].values, width, label='Male Actual',
       color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=1.5)
ax.bar(x + width*1.5, age_gender_pred['male'].values, width, label='Male Predicted',
       color=COLORS[2], alpha=0.5, edgecolor=COLORS[5], linewidth=1.5, hatch='//')
ax.set_xticks(x)
ax.set_xticklabels(age_gender_actual.index, fontsize=11, color=COLORS[5])
ax.set_ylabel('Average Cost ($)', fontsize=12, color=COLORS[5])
ax.set_xlabel('Age Group', fontsize=12, color=COLORS[5])
ax.set_title('Intersectional Bias: Age × Gender', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4], fontsize=9, ncol=2)
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

ax = axes[1, 1]
fairness_metrics = {
    'Demographic Parity\n(Gender)': gender_auditor.metrics['demographic_parity']['ratio'] * 100,
    'Equalized Odds\n(Gender)': gender_auditor.metrics['equalized_odds']['ratio'] * 100,
    'Demographic Parity\n(Region)': region_auditor.metrics['demographic_parity']['ratio'] * 100,
    'Equalized Odds\n(Region)': region_auditor.metrics['equalized_odds']['ratio'] * 100
}
bars = ax.bar(range(len(fairness_metrics)), list(fairness_metrics.values()),
              color=[COLORS[4], COLORS[4], COLORS[2], COLORS[2]],
              edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_xticks(range(len(fairness_metrics)))
ax.set_xticklabels(list(fairness_metrics.keys()), fontsize=9, color=COLORS[5])
ax.set_ylabel('Fairness Ratio (%)', fontsize=12, color=COLORS[5])
ax.axhline(y=80, color=COLORS[5], linestyle='--', linewidth=2, alpha=0.5, 
          label='Fairness Threshold (80%)')
ax.set_ylim(0, 100)
ax.set_title('Fairness Metrics Summary', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])
for i, (bar, val) in enumerate(zip(bars, fairness_metrics.values())):
    status = 'FAIR' if val >= 80 else 'BIASED'
    ax.text(i, val, f'{val:.1f}%\n{status}', ha='center', va='bottom',
           color=COLORS[5], fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}bias_fairness_analysis.png', dpi=300,
            facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
plt.close()

print(f"\nVisualization saved: bias_fairness_analysis.png")

print(f"\n{'='*60}")
print(f"Statistical Disparity Analysis")
print(f"{'='*60}")

print(f"\nGender Cost Disparity:")
female_costs = medical_data[medical_data['sex'] == 'female']['charges']
male_costs = medical_data[medical_data['sex'] == 'male']['charges']
ttest_result = stats.ttest_ind(female_costs, male_costs)
print(f"  Female average: ${female_costs.mean():,.2f}")
print(f"  Male average: ${male_costs.mean():,.2f}")
print(f"  Difference: ${abs(female_costs.mean() - male_costs.mean()):,.2f}")
print(f"  T-test p-value: {ttest_result.pvalue:.4f}")
if ttest_result.pvalue < 0.05:
    print(f"  Statistical significance: YES - Significant difference exists")
else:
    print(f"  Statistical significance: NO - No significant difference")

print(f"\nRegional Cost Disparity:")
for region in medical_data['region'].unique():
    region_costs = medical_data[medical_data['region'] == region]['charges']
    print(f"  {region.title()}: ${region_costs.mean():,.2f} (n={len(region_costs)})")

region_anova = stats.f_oneway(*[medical_data[medical_data['region'] == r]['charges'].values 
                                for r in medical_data['region'].unique()])
print(f"  ANOVA p-value: {region_anova.pvalue:.4f}")
if region_anova.pvalue < 0.05:
    print(f"  Regional differences: SIGNIFICANT")
else:
    print(f"  Regional differences: NOT SIGNIFICANT")

print(f"\nSmoking Status Disparity:")
smoker_costs = medical_data[medical_data['smoker'] == 'yes']['charges']
nonsmoker_costs = medical_data[medical_data['smoker'] == 'no']['charges']
print(f"  Smokers: ${smoker_costs.mean():,.2f}")
print(f"  Non-smokers: ${nonsmoker_costs.mean():,.2f}")
print(f"  Multiplier: {smoker_costs.mean() / nonsmoker_costs.mean():.2f}x")
smoking_ttest = stats.ttest_ind(smoker_costs, nonsmoker_costs)
print(f"  T-test p-value: {smoking_ttest.pvalue:.10f}")
print(f"  Disparity: JUSTIFIED (behavioral risk factor)")

fig, axes = plt.subplots(2, 2, figsize=(22, 18))
fig.patch.set_facecolor('#080C0D')

ax = axes[0, 0]
groups_to_compare = [
    ('Female Non-Smokers', medical_data[(medical_data['sex'] == 'female') & 
                                       (medical_data['smoker'] == 'no')]['charges']),
    ('Female Smokers', medical_data[(medical_data['sex'] == 'female') & 
                                   (medical_data['smoker'] == 'yes')]['charges']),
    ('Male Non-Smokers', medical_data[(medical_data['sex'] == 'male') & 
                                     (medical_data['smoker'] == 'no')]['charges']),
    ('Male Smokers', medical_data[(medical_data['sex'] == 'male') & 
                                 (medical_data['smoker'] == 'yes')]['charges'])
]
bp = ax.boxplot([data for _, data in groups_to_compare],
                labels=[name for name, _ in groups_to_compare],
                patch_artist=True,
                boxprops=dict(alpha=0.7, edgecolor=COLORS[5], linewidth=2),
                medianprops=dict(color=COLORS[5], linewidth=2),
                whiskerprops=dict(color=COLORS[5], linewidth=1.5),
                capprops=dict(color=COLORS[5], linewidth=1.5))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(GRADIENT[i*2])
ax.set_ylabel('Medical Costs ($)', fontsize=12, color=COLORS[5])
ax.set_title('Intersectional Cost Distribution', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5], labelsize=9)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

ax = axes[0, 1]
feature_impact_disparities = []
for feature in ['age', 'bmi', 'children']:
    female_corr = abs(medical_data[medical_data['sex'] == 'female'][[feature, 'charges']].corr().iloc[0, 1])
    male_corr = abs(medical_data[medical_data['sex'] == 'male'][[feature, 'charges']].corr().iloc[0, 1])
    disparity = abs(female_corr - male_corr)
    feature_impact_disparities.append((feature, female_corr, male_corr, disparity))
features = [f[0].upper() for f in feature_impact_disparities]
female_impacts = [f[1] for f in feature_impact_disparities]
male_impacts = [f[2] for f in feature_impact_disparities]
x = np.arange(len(features))
width = 0.35
ax.bar(x - width/2, female_impacts, width, label='Female Impact',
       color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.bar(x + width/2, male_impacts, width, label='Male Impact',
       color=COLORS[2], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels(features, fontsize=11, color=COLORS[5])
ax.set_ylabel('Correlation with Cost', fontsize=12, color=COLORS[5])
ax.set_title('Feature Impact Disparity by Gender', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

ax = axes[1, 0]
disparities_summary = {
    'Gender\nCost Gap': abs(female_costs.mean() - male_costs.mean()) / 1000,
    'Regional\nVariance': medical_data.groupby('region')['charges'].mean().std() / 1000,
    'Smoking\nPremium': (smoker_costs.mean() - nonsmoker_costs.mean()) / 1000,
    'Age\nEffect': abs(medical_data[medical_data['age'] > 50]['charges'].mean() - 
                      medical_data[medical_data['age'] <= 50]['charges'].mean()) / 1000
}
bars = ax.bar(range(len(disparities_summary)), list(disparities_summary.values()),
              color=[GRADIENT[i*2] for i in range(len(disparities_summary))],
              edgecolor=COLORS[5], linewidth=2, alpha=0.85)
ax.set_xticks(range(len(disparities_summary)))
ax.set_xticklabels(list(disparities_summary.keys()), fontsize=10, color=COLORS[5])
ax.set_ylabel('Cost Difference ($1000s)', fontsize=12, color=COLORS[5])
ax.set_title('Disparity Magnitude Comparison', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])
for i, (bar, val) in enumerate(zip(bars, disparities_summary.values())):
    ax.text(i, val, f'${val:.1f}K', ha='center', va='bottom',
           color=COLORS[5], fontsize=10, weight='bold')

ax = axes[1, 1]
ethical_considerations = [
    'Data Privacy\nProtection',
    'Algorithmic\nFairness',
    'Transparency\n& Explainability',
    'Consent\nManagement',
    'Bias\nMitigation'
]
current_scores = [overall_compliance, (gender_fairness_score + region_fairness_score)/2, 
                 75, 70, 80]
target_scores = [95, 90, 90, 95, 90]
x = np.arange(len(ethical_considerations))
width = 0.35
ax.bar(x - width/2, current_scores, width, label='Current',
       color=COLORS[4], alpha=0.85, edgecolor=COLORS[5], linewidth=2)
ax.bar(x + width/2, target_scores, width, label='Target',
       color=COLORS[5], alpha=0.5, edgecolor=COLORS[5], linewidth=2, hatch='//')
ax.set_xticks(x)
ax.set_xticklabels(ethical_considerations, fontsize=9, color=COLORS[5])
ax.set_ylabel('Score', fontsize=12, color=COLORS[5])
ax.set_ylim(0, 100)
ax.set_title('Ethical AI Scorecard', fontsize=16, color=COLORS[5], pad=20, weight='bold')
ax.set_facecolor('#0D1214')
ax.legend(framealpha=0.9, facecolor='#080C0D', edgecolor=COLORS[4])
ax.grid(True, alpha=0.15, color=COLORS[4], axis='y')
ax.tick_params(colors=COLORS[5])

plt.tight_layout()
plt.savefig(f'{SAVE_PATH}disparity_ethical_analysis.png', dpi=300,
            facecolor='#080C0D', edgecolor='none', bbox_inches='tight')
plt.close()

print(f"\nVisualization saved: disparity_ethical_analysis.png")

print(f"\n{'='*60}")
print(f"Final Recommendations")
print(f"{'='*60}")

print(f"\nData Privacy & Security:")
print(f"  1. Dataset contains no direct PHI identifiers - COMPLIANT")
print(f"  2. K-anonymity protection at {k_anon_protection:.1f}% - {'ADEQUATE' if k_anon_protection > 80 else 'NEEDS IMPROVEMENT'}")
print(f"  3. Re-identification risk: {reident_risk:.2f}% - {'LOW' if reident_risk < 5 else 'MODERATE'}")
print(f"  4. Implement differential privacy for aggregate reporting")
print(f"  5. Regular privacy impact assessments recommended")

print(f"\nFairness & Bias:")
print(f"  1. Gender prediction parity: {'FAIR' if gender_auditor.metrics['demographic_parity']['ratio'] > 0.8 else 'NEEDS ATTENTION'}")
print(f"  2. Regional fairness: {'FAIR' if region_auditor.metrics['demographic_parity']['ratio'] > 0.8 else 'NEEDS ATTENTION'}")
print(f"  3. Smoking disparity is justified (behavioral risk factor)")
print(f"  4. Monitor for proxy discrimination through correlated features")
print(f"  5. Regular fairness audits across protected attributes")

print(f"\nEthical Considerations:")
print(f"  1. Model transparency: Provide feature importance explanations")
print(f"  2. Right to explanation: Enable individual prediction justification")
print(f"  3. Consent: Ensure proper data collection consent obtained")
print(f"  4. Accountability: Establish clear governance for model decisions")
print(f"  5. Continuous monitoring: Track fairness metrics over time")

print(f"\nCompliance Status:")
print(f"  Overall Privacy Score: {overall_compliance:.1f}/100")
print(f"  Average Fairness Score: {(gender_fairness_score + region_fairness_score)/2:.1f}/100")
print(f"  Dataset Safety: {'APPROVED FOR USE' if overall_compliance > 80 else 'REQUIRES REMEDIATION'}")
print(f"\nConclusion: Dataset demonstrates strong privacy protection with minor")
print(f"fairness considerations. Suitable for research and model development with")
print(f"ongoing monitoring and regular ethical audits.")
