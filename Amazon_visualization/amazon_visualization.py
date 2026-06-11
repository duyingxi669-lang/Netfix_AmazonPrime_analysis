import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Generating Amazon Prime Data Visualization Charts")
print("=" * 60)

# ========== 读取所有数据 ==========
movies_tvshows = pd.read_csv('amazon_movies_tvshows.csv')
released_movie = pd.read_csv('amazon_released_movie.csv')
released_tv_show = pd.read_csv('amazon_released_tv_show.csv')
country_data = pd.read_csv('amazon_country_data.csv')
data_q2q3 = pd.read_csv('amazon_data_q2q3.csv')
movie_duration = pd.read_csv('amazon_movie_duration.csv')
rating_tvshow = pd.read_csv('amazon_rating_tvshow.csv')
rating_movie = pd.read_csv('amazon_rating_movie.csv')
categories = pd.read_csv('amazon_categories.csv')
classification = pd.read_csv('amazon_classification.csv')

# 演员文件（8个国家）
actors_files = {
    'US': 'amazon_actors_US.csv',
    'IN': 'amazon_actors_IN.csv', 
    'GB': 'amazon_actors_GB.csv',
    'CA': 'amazon_actors_CA.csv',
    'FR': 'amazon_actors_FR.csv',
    'JP': 'amazon_actors_JP.csv',
    'DE': 'amazon_actors_DE.csv',
    'CN': 'amazon_actors_CN.csv'
}

print("✓ All data loaded successfully")

# ========== Chart 1: Movies vs TV Shows (Pie Chart) ==========
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#4e8d7c', '#ea97ad']
ax.pie(movies_tvshows['count'], labels=movies_tvshows['type'], 
       autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 14})
ax.set_title('Amazon Prime: Movies vs TV Shows', fontsize=16, fontweight='bold')
plt.savefig('amazon_fig1_pie.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 1: Movies vs TV Shows")

# ========== Chart 2: Content Released Over Years ==========
fig, ax = plt.subplots(figsize=(14, 7))

# 清理数据，只保留有效年份（1900-2023）
movie_clean = released_movie[pd.to_numeric(released_movie['release_year'], errors='coerce').notna()]
tv_clean = released_tv_show[pd.to_numeric(released_tv_show['release_year'], errors='coerce').notna()]
movie_clean = movie_clean[(movie_clean['release_year'] >= 1920) & (movie_clean['release_year'] <= 2023)]
tv_clean = tv_clean[(tv_clean['release_year'] >= 1920) & (tv_clean['release_year'] <= 2023)]

ax.plot(movie_clean['release_year'], movie_clean['count'], 'o-', 
        color='#4e8d7c', linewidth=2, markersize=3, label='Movies')
ax.plot(tv_clean['release_year'], tv_clean['count'], 's-', 
        color='#ea97ad', linewidth=2, markersize=3, label='TV Shows')
ax.set_xlabel('Release Year', fontsize=12)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.set_title('Amazon Prime: Content Released Over the Years', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig('amazon_fig2_released_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 2: Released trend")

# ========== Chart 3: Top Countries with Most Content ==========
country_sorted = country_data.nlargest(15, 'count').sort_values('count', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(country_sorted['country'], country_sorted['count'], color='#4e8d7c', alpha=0.8)
ax.set_xlabel('Number of Titles', fontsize=12)
ax.set_title('Amazon Prime: Top 15 Countries with Most Content', fontsize=14, fontweight='bold')
for i, (country, count) in enumerate(zip(country_sorted['country'], country_sorted['count'])):
    ax.text(count + 20, i, str(int(count)), va='center', fontsize=10)
plt.tight_layout()
plt.savefig('amazon_fig3_country_total.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 3: Country ranking")

# ========== Chart 4: Movies vs TV Shows by Country ==========
fig, ax = plt.subplots(figsize=(12, 8))
countries = ['US', 'IN', 'GB', 'CA', 'FR', 'JP', 'DE', 'CN', 'AU', 'IT']
country_names = ['US', 'IN', 'GB', 'CA', 'FR', 'JP', 'DE', 'CN', 'AU', 'IT']
# 重新排序数据
data_q2q3_filtered = data_q2q3[data_q2q3['country'].isin(countries)]

ax.barh(data_q2q3_filtered['country'], data_q2q3_filtered['MOVIE'], 
        color='#4e8d7c', label='Movies')
ax.barh(data_q2q3_filtered['country'], data_q2q3_filtered['SHOW'], 
        left=data_q2q3_filtered['MOVIE'], color='#ea97ad', label='TV Shows')
ax.set_xlabel('Percentage', fontsize=12)
ax.set_title('Amazon Prime: Movies vs TV Shows by Country', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xlim(0, 1)
for i, (movie, tv) in enumerate(zip(data_q2q3_filtered['MOVIE'], data_q2q3_filtered['SHOW'])):
    if movie > 0.05:
        ax.text(movie/2, i, f'{movie*100:.0f}%', ha='center', va='center', fontsize=9, color='white')
    if tv > 0.05:
        ax.text(movie + tv/2, i, f'{tv*100:.0f}%', ha='center', va='center', fontsize=9, color='white')
plt.tight_layout()
plt.savefig('amazon_fig4_country_split.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 4: Country split")

# ========== Chart 5: Movie Duration Distribution ==========
durations = movie_duration['duration'].dropna()
durations = durations[durations > 0]

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(durations, bins=40, density=True, alpha=0.6, color='#4e8d7c', edgecolor='white')
mu, std = durations.mean(), durations.std()
x = np.linspace(durations.min(), durations.max(), 200)
y = stats.norm.pdf(x, mu, std)
ax.plot(x, y, 'r-', linewidth=2.5, label=f'Normal Dist (μ={mu:.1f}, σ={std:.1f})')
ax.set_xlabel('Duration (minutes)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Amazon Prime: Movie Duration Distribution', fontsize=14, fontweight='bold')
ax.axvline(mu, color='red', linestyle='--', alpha=0.7)
ax.text(mu + 5, stats.norm.pdf(mu, mu, std) * 0.9, f'Mean: {mu:.1f} min', fontsize=10, color='red')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('amazon_fig5_duration.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 5: Duration distribution")

# ========== Chart 6: Rating Distribution ==========
ratings_order = ['<5', '5-6', '6-7', '7-8', '8-9', '9-10']
rating_tvshow_sorted = rating_tvshow.set_index('index').reindex(ratings_order).reset_index()
rating_movie_sorted = rating_movie.set_index('index').reindex(ratings_order).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(ratings_order))
width = 0.35
ax.bar([i - width/2 for i in x], rating_tvshow_sorted['count_tvshow'], 
       width, label='TV Shows', color='#ea97ad')
ax.bar([i + width/2 for i in x], rating_movie_sorted['count_movie'], 
       width, label='Movies', color='#4e8d7c')
ax.set_xlabel('IMDB Score Range', fontsize=12)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.set_title('Amazon Prime: Rating Distribution (IMDB)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(ratings_order)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('amazon_fig6_rating.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 6: Rating distribution")

# ========== Chart 7: Top Content Categories ==========
categories_sorted = categories.nlargest(15, 'Count').sort_values('Count', ascending=True)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(categories_sorted['Category'], categories_sorted['Count'], color='#ea97ad', alpha=0.8)
ax.set_xlabel('Number of Titles', fontsize=12)
ax.set_title('Amazon Prime: Top 15 Content Categories', fontsize=14, fontweight='bold')
for i, (cat, count) in enumerate(zip(categories_sorted['Category'], categories_sorted['Count'])):
    ax.text(count + 50, i, str(int(count)), va='center', fontsize=9)
plt.tight_layout()
plt.savefig('amazon_fig7_categories.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 7: Content categories")

# ========== Chart 8: Feature Classification Accuracy ==========
classification_sorted = classification.sort_values('Accuracy', ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(classification_sorted['Feature Column'], classification_sorted['Accuracy'], 
                color='#4e8d7c', alpha=0.8)
ax.set_xlabel('Accuracy', fontsize=12)
ax.set_title('Amazon Prime: Classification Accuracy by Feature', fontsize=14, fontweight='bold')
ax.set_xlim(0.85, 0.95)  # Amazon数据准确率更高
max_acc = classification_sorted['Accuracy'].max()
for i, (feature, acc) in enumerate(zip(classification_sorted['Feature Column'], classification_sorted['Accuracy'])):
    if acc == max_acc:
        bars[i].set_color('#ea97ad')
        ax.text(acc + 0.003, i, f'★ Best: {acc:.4f}', va='center', fontsize=9, fontweight='bold')
    else:
        ax.text(acc + 0.003, i, f'{acc:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig('amazon_fig8_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 8: Feature accuracy")

# ========== Chart 9-10: Popular Actors (Top 8 Countries) ==========
top_countries = ['US', 'IN', 'GB', 'CA', 'FR', 'JP', 'DE', 'CN']
fig, axes = plt.subplots(2, 4, figsize=(20, 12))
axes = axes.flatten()

for idx, country in enumerate(top_countries):
    if country in actors_files:
        actors_df = pd.read_csv(actors_files[country])
        top10 = actors_df.head(10).sort_values('count', ascending=True)
        ax = axes[idx]
        ax.barh(top10['actor'], top10['count'], color='#ea97ad')
        ax.set_title(f'Top Actors - {country}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Appearances', fontsize=10)
        ax.tick_params(axis='y', labelsize=8)

plt.suptitle('Amazon Prime: Most Popular Actors by Country', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('amazon_fig9_actors.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 9-10: Popular actors")

print("=" * 60)
print("✅ ALL 10 AMAZON PRIME CHARTS GENERATED!")
print("=" * 60)
print("Output files:")
print("  amazon_fig1_pie.png - Movies vs TV Shows")
print("  amazon_fig2_released_trend.png - Released trend")
print("  amazon_fig3_country_total.png - Country ranking")
print("  amazon_fig4_country_split.png - Country split")
print("  amazon_fig5_duration.png - Duration distribution")
print("  amazon_fig6_rating.png - Rating distribution")
print("  amazon_fig7_categories.png - Content categories")
print("  amazon_fig8_accuracy.png - Feature accuracy")
print("  amazon_fig9_actors.png - Popular actors")