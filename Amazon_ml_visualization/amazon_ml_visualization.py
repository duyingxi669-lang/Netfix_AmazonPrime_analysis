import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Generating Amazon Prime ML Analysis Charts")
print("=" * 60)

# ========== 读取数据 ==========
try:
    genre_avg = pd.read_csv('amazon_genre_avg_score.csv')
    print("✓ Loaded genre_avg_score.csv")
except:
    genre_avg = pd.read_csv('output/amazon_genre_avg_score.csv')

try:
    decade_avg = pd.read_csv('amazon_decade_avg_score.csv')
    print("✓ Loaded decade_avg_score.csv")
except:
    decade_avg = pd.read_csv('output/amazon_decade_avg_score.csv')

try:
    type_compare = pd.read_csv('amazon_type_score_compare.csv')
    print("✓ Loaded type_score_compare.csv")
except:
    type_compare = pd.read_csv('output/amazon_type_score_compare.csv')

try:
    country_avg = pd.read_csv('amazon_country_avg_score.csv')
    print("✓ Loaded country_avg_score.csv")
except:
    country_avg = pd.read_csv('output/amazon_country_avg_score.csv')

try:
    regression_compare = pd.read_csv('amazon_regression_compare.csv')
    print("✓ Loaded regression_compare.csv")
except:
    regression_compare = pd.read_csv('output/amazon_regression_compare.csv')

try:
    rf_importance = pd.read_csv('amazon_rf_importance.csv')
    print("✓ Loaded rf_importance.csv")
except:
    rf_importance = pd.read_csv('output/amazon_rf_importance.csv')

try:
    gbt_importance = pd.read_csv('amazon_gbt_importance.csv')
    print("✓ Loaded gbt_importance.csv")
except:
    gbt_importance = pd.read_csv('output/amazon_gbt_importance.csv')

try:
    classification_compare = pd.read_csv('amazon_classification_compare.csv')
    print("✓ Loaded classification_compare.csv")
except:
    classification_compare = pd.read_csv('output/amazon_classification_compare.csv')

try:
    clustered = pd.read_csv('amazon_clustered.csv')
    print("✓ Loaded clustered.csv")
except:
    clustered = pd.read_csv('output/amazon_clustered.csv')

print("\n✓ All data loaded successfully")

# ========== Chart 1: Genre Average Score (Top 15) ==========
genre_sorted = genre_avg.nlargest(15, 'avg_score').sort_values('avg_score', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(genre_sorted)))
ax.barh(genre_sorted['genre'], genre_sorted['avg_score'], color=colors, alpha=0.8)
ax.set_xlabel('Average IMDB Score', fontsize=12)
ax.set_title('Top 15 Genres by Average IMDB Score', fontsize=14, fontweight='bold')
ax.set_xlim(5.5, 7.5)
for i, (genre, score, count) in enumerate(zip(genre_sorted['genre'], genre_sorted['avg_score'], genre_sorted['count'])):
    ax.text(score + 0.05, i, f'{score:.2f} ({count:,})', va='center', fontsize=9)
ax.axvline(6.0, color='red', linestyle='--', alpha=0.5, label='Global Average (~6.0)')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('amazon_ml_fig1_genre_score.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 1: Genre average score")

# ========== Chart 2: Decade Average Score Trend ==========
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(decade_avg['decade'], decade_avg['avg_score'], 'o-', color='#4e8d7c', linewidth=2.5, markersize=8)
ax.fill_between(decade_avg['decade'], 5.5, decade_avg['avg_score'], alpha=0.3, color='#4e8d7c')
ax.set_xlabel('Decade', fontsize=12)
ax.set_ylabel('Average IMDB Score', fontsize=12)
ax.set_title('IMDB Score Trend by Decade', fontsize=14, fontweight='bold')
ax.set_ylim(5.5, 6.5)
ax.grid(True, alpha=0.3)
# Add global average line
ax.axhline(6.0, color='red', linestyle='--', alpha=0.5, label='Global Average')
ax.legend()
plt.tight_layout()
plt.savefig('amazon_ml_fig2_decade_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 2: Decade trend")

# ========== Chart 3: Movies vs TV Shows Score Comparison ==========
fig, ax = plt.subplots(figsize=(10, 6))
types = type_compare['type']
avg_scores = type_compare['avg_score']
std_scores = type_compare['std_score']

x = range(len(types))
bars = ax.bar(x, avg_scores, yerr=std_scores, capsize=10, color=['#4e8d7c', '#ea97ad'], alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(types)
ax.set_ylabel('Average IMDB Score', fontsize=12)
ax.set_title('Movies vs TV Shows: IMDB Score Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(5, 8)
for bar, score in zip(bars, avg_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{score:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('amazon_ml_fig3_type_score.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 3: Movies vs TV Shows score")

# ========== Chart 4: Top Countries by Average Score ==========
country_sorted = country_avg.nlargest(15, 'avg_score').sort_values('avg_score', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(country_sorted)))
ax.barh(country_sorted['country'], country_sorted['avg_score'], color=colors, alpha=0.8)
ax.set_xlabel('Average IMDB Score', fontsize=12)
ax.set_title('Top 15 Countries by Average IMDB Score', fontsize=14, fontweight='bold')
ax.set_xlim(5.5, 7.0)
for i, (country, score, count) in enumerate(zip(country_sorted['country'], country_sorted['avg_score'], country_sorted['count'])):
    ax.text(score + 0.05, i, f'{score:.2f} (n={count})', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('amazon_ml_fig4_country_score.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 4: Country average score")

# ========== Chart 5: Regression Model Comparison ==========
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(regression_compare))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], regression_compare['RMSE'], width, label='RMSE', color='#4e8d7c', alpha=0.8)
ax2 = ax.twinx()
bars2 = ax2.bar([i + width/2 for i in x], regression_compare['R2'], width, label='R²', color='#ea97ad', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(regression_compare['Model'])
ax.set_ylabel('RMSE (lower is better)', fontsize=12, color='#4e8d7c')
ax2.set_ylabel('R² (higher is better)', fontsize=12, color='#ea97ad')
ax.set_title('Regression Model Performance: IMDB Score Prediction', fontsize=14, fontweight='bold')
ax.tick_params(axis='y', labelcolor='#4e8d7c')
ax2.tick_params(axis='y', labelcolor='#ea97ad')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('amazon_ml_fig5_regression_compare.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 5: Regression model comparison")

# ========== Chart 6: Feature Importance (Top 15) ==========
rf_top15 = rf_importance.nlargest(15, 'Importance').sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(rf_top15)))
ax.barh(rf_top15['Feature'], rf_top15['Importance'], color=colors, alpha=0.8)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Random Forest Feature Importance (IMDB Score Prediction)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('amazon_ml_fig6_rf_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 6: Random Forest feature importance")

# ========== Chart 7: GBT Feature Importance ==========
gbt_top15 = gbt_importance.nlargest(15, 'Importance').sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(gbt_top15)))
ax.barh(gbt_top15['Feature'], gbt_top15['Importance'], color=colors, alpha=0.8)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Gradient Boosted Trees Feature Importance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('amazon_ml_fig7_gbt_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 7: GBT feature importance")

# ========== Chart 8: Classification Model Comparison ==========
fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(classification_compare))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], classification_compare['Accuracy'], width, label='Accuracy', color='#4e8d7c', alpha=0.8)
bars2 = ax.bar([i + width/2 for i in x], classification_compare['F1'], width, label='F1 Score', color='#ea97ad', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(classification_compare['Model'], rotation=15, ha='right')
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Classification Model Performance (Movie vs TV Show)', fontsize=14, fontweight='bold')
ax.set_ylim(0.95, 0.98)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

for i, (acc, f1) in enumerate(zip(classification_compare['Accuracy'], classification_compare['F1'])):
    ax.text(i - width/2, acc + 0.001, f'{acc:.4f}', ha='center', fontsize=8)
    ax.text(i + width/2, f1 + 0.001, f'{f1:.4f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('amazon_ml_fig8_classification_compare.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 8: Classification model comparison")

# ========== Chart 9: Clustering Analysis ==========
cluster_summary = clustered.groupby('prediction').agg({
    'imdb_score': 'mean',
    'runtime': 'mean',
    'release_year': 'mean',
    'actor_count': 'mean'
}).reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
x = cluster_summary['prediction']
width = 0.2
ax.bar(x - 1.5*width, cluster_summary['imdb_score'], width, label='Avg IMDB Score', color='#4e8d7c')
ax2 = ax.twinx()
ax2.bar(x - 0.5*width, cluster_summary['runtime'], width, label='Avg Runtime (min)', color='#ea97ad')
ax2.bar(x + 0.5*width, cluster_summary['actor_count']/10, width, label='Actor Count /10', color='#ffb347')
ax.bar(x + 1.5*width, (cluster_summary['release_year']-1900)/10, width, label='(Year-1900)/10', color='#87ceeb')

ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('IMDB Score', fontsize=12, color='#4e8d7c')
ax2.set_ylabel('Runtime / Actor Count', fontsize=12, color='#ea97ad')
ax.set_title('Cluster Characteristics (K-Means)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.tick_params(axis='y', labelcolor='#4e8d7c')
ax2.tick_params(axis='y', labelcolor='#ea97ad')
plt.tight_layout()
plt.savefig('amazon_ml_fig9_clustering.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 9: Clustering analysis")

# ========== Chart 10: Cluster Size Distribution ==========
cluster_sizes = clustered['prediction'].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
bars = ax.bar(cluster_sizes.index, cluster_sizes.values, color=colors, edgecolor='white', linewidth=2)
ax.set_xlabel('Cluster', fontsize=12)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.set_title('Cluster Size Distribution (K-Means)', fontsize=14, fontweight='bold')
for bar, size in zip(bars, cluster_sizes.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, str(size), ha='center', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('amazon_ml_fig10_cluster_size.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 10: Cluster size distribution")

print("=" * 60)
print("✅ ALL 10 ML CHARTS GENERATED SUCCESSFULLY!")
print("=" * 60)
print("Output files:")
print("  amazon_ml_fig1_genre_score.png - Top genres by score")
print("  amazon_ml_fig2_decade_trend.png - Score trend by decade")
print("  amazon_ml_fig3_type_score.png - Movies vs TV shows")
print("  amazon_ml_fig4_country_score.png - Top countries by score")
print("  amazon_ml_fig5_regression_compare.png - Regression models")
print("  amazon_ml_fig6_rf_importance.png - RF feature importance")
print("  amazon_ml_fig7_gbt_importance.png - GBT feature importance")
print("  amazon_ml_fig8_classification_compare.png - Classification models")
print("  amazon_ml_fig9_clustering.png - Cluster characteristics")
print("  amazon_ml_fig10_cluster_size.png - Cluster size distribution")