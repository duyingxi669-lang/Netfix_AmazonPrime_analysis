import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("Generating Netflix Data Visualization Charts (Corrected Data)")
print("=" * 60)

# Load data
movies_tvshows = pd.read_csv('movies_tvshows.csv')
added_movie = pd.read_csv('added_movie.csv')
added_tv_show = pd.read_csv('added_tv_show.csv')
released_movie = pd.read_csv('released_movie.csv')
released_tv_show = pd.read_csv('released_tv_show.csv')
country_data = pd.read_csv('country_data.csv')
data_q2q3 = pd.read_csv('data_q2q3.csv')
movie_duration = pd.read_csv('movie_duration.csv')
rating_tvshow = pd.read_csv('rating_tvshow.csv')
rating_movie = pd.read_csv('rating_movie.csv')
categories = pd.read_csv('categories.csv')
classification = pd.read_csv('classification.csv')

actors_files = {
    'United States': 'actors_United_States.csv',
    'India': 'actors_India.csv',
    'United Kingdom': 'actors_United_Kingdom.csv',
    'Japan': 'actors_Japan.csv',
    'South Korea': 'actors_South_Korea.csv',
    'Canada': 'actors_Canada.csv'
}

print("✓ All data loaded successfully")

# Chart 1
fig, ax = plt.subplots(figsize=(8, 8))
colors = ['#4e8d7c', '#ea97ad']
ax.pie(movies_tvshows['count'], labels=movies_tvshows['type'], autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Netflix: Proportion of Movies and TV Shows', fontsize=16, fontweight='bold')
plt.savefig('netflix_fig1_pie.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 1")

# Chart 2
added_movie['year'] = pd.to_numeric(added_movie['year_added'], errors='coerce')
added_tv_show['year'] = pd.to_numeric(added_tv_show['year_added'], errors='coerce')
movie_added = added_movie.dropna(subset=['year'])
tv_added = added_tv_show.dropna(subset=['year'])
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(movie_added['year'], movie_added['count'], 'o-', color='#4e8d7c', label='Movies')
ax.plot(tv_added['year'], tv_added['count'], 's-', color='#ea97ad', label='TV Shows')
ax.set_xlabel('Year Added')
ax.set_ylabel('Number of Titles')
ax.set_title('Content Added to Netflix Over the Years')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('netflix_fig2_added_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 2")

# Chart 3
released_movie['year'] = pd.to_numeric(released_movie['release_year'], errors='coerce')
released_tv_show['year'] = pd.to_numeric(released_tv_show['release_year'], errors='coerce')
movie_released = released_movie.dropna(subset=['year'])
tv_released = released_tv_show.dropna(subset=['year'])
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(movie_released['year'], movie_released['count'], 'o-', color='#4e8d7c', label='Movies')
ax.plot(tv_released['year'], tv_released['count'], 's-', color='#ea97ad', label='TV Shows')
ax.set_xlabel('Release Year')
ax.set_ylabel('Number of Titles')
ax.set_title('Content Released Over the Years')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('netflix_fig3_released_trend.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 3")

# Chart 4
country_counts = country_data.copy()
country_counts['count'] = pd.to_numeric(country_counts['count'], errors='coerce')
top_countries = country_counts.nlargest(15, 'count').sort_values('count', ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top_countries['country'], top_countries['count'], color='#4e8d7c')
ax.set_xlabel('Number of Titles')
ax.set_title('Top 15 Countries with Most Content on Netflix')
for i, (country, count) in enumerate(zip(top_countries['country'], top_countries['count'])):
    ax.text(count + 10, i, str(int(count)), va='center')
plt.tight_layout()
plt.savefig('netflix_fig4_country_total.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 4")

# Chart 5
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(data_q2q3['country'], data_q2q3['Movie'], color='#4e8d7c', label='Movies')
ax.barh(data_q2q3['country'], data_q2q3['TV Show'], left=data_q2q3['Movie'], color='#ea97ad', label='TV Shows')
ax.set_xlabel('Percentage')
ax.set_title('Top 10 Countries: Movies vs TV Shows Split')
ax.legend()
ax.set_xlim(0, 1)
for i, (movie, tv) in enumerate(zip(data_q2q3['Movie'], data_q2q3['TV Show'])):
    if movie > 0.05:
        ax.text(movie/2, i, f'{movie*100:.0f}%', ha='center', va='center', color='white')
    if tv > 0.05:
        ax.text(movie + tv/2, i, f'{tv*100:.0f}%', ha='center', va='center', color='white')
plt.tight_layout()
plt.savefig('netflix_fig5_country_split.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 5")

# Chart 6
durations = pd.to_numeric(movie_duration['duration'], errors='coerce').dropna()
durations = durations[durations > 0]
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(durations, bins=40, density=True, alpha=0.6, color='#4e8d7c', edgecolor='white')
mu, std = durations.mean(), durations.std()
x = np.linspace(durations.min(), durations.max(), 200)
y = stats.norm.pdf(x, mu, std)
ax.plot(x, y, 'r-', linewidth=2.5, label=f'Normal Dist (μ={mu:.1f}, σ={std:.1f})')
ax.set_xlabel('Duration (minutes)')
ax.set_ylabel('Density')
ax.set_title('Movie Duration Distribution with Normal Curve')
ax.axvline(mu, color='red', linestyle='--')
ax.text(mu + 5, stats.norm.pdf(mu, mu, std) * 0.9, f'Mean: {mu:.1f} min', color='red')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('netflix_fig6_duration.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 6")

# Chart 7
ratings_order = ['TV-MA', 'TV-14', 'TV-PG', 'R', 'PG-13', 'PG', 'TV-Y7', 'TV-Y', 'TV-G', 'NR', 'G', 'NC-17']
tv_clean = rating_tvshow.copy()
tv_clean['index'] = tv_clean['index'].str.strip()
tv_clean = tv_clean[tv_clean['index'].isin(ratings_order)]

movie_clean = rating_movie.copy()
movie_clean['index'] = movie_clean['index'].str.strip()
movie_clean = movie_clean[movie_clean['index'].isin(ratings_order)]

tv_counts = []
movie_counts = []
for r in ratings_order:
    tv_val = tv_clean[tv_clean['index'] == r]['count_tvshow'].values
    movie_val = movie_clean[movie_clean['index'] == r]['count_movie'].values
    tv_counts.append(tv_val[0] if len(tv_val) > 0 else 0)
    movie_counts.append(movie_val[0] if len(movie_val) > 0 else 0)

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(ratings_order))
width = 0.35
ax.bar(x - width/2, tv_counts, width, label='TV Shows', color='#ea97ad')
ax.bar(x + width/2, movie_counts, width, label='Movies', color='#4e8d7c')
ax.set_xlabel('Rating', fontsize=12)
ax.set_ylabel('Number of Titles', fontsize=12)
ax.set_title('Rating Distribution: Movies vs TV Shows', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(ratings_order, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('netflix_fig7_rating.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 7 (fixed)")

# Chart 8
categories_sorted = categories.nlargest(20, 'Count').sort_values('Count', ascending=True)
fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(categories_sorted['Category'], categories_sorted['Count'], color='#ea97ad')
ax.set_xlabel('Number of Titles')
ax.set_title('Top 20 Content Categories on Netflix')
for i, (cat, count) in enumerate(zip(categories_sorted['Category'], categories_sorted['Count'])):
    ax.text(count + 20, i, str(int(count)), va='center')
plt.tight_layout()
plt.savefig('netflix_fig8_categories.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 8")

# Chart 9
classification_sorted = classification.sort_values('Accuracy', ascending=True)
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(classification_sorted['Feature Column'], classification_sorted['Accuracy'], color='#4e8d7c')
ax.set_xlabel('Accuracy')
ax.set_title('Classification Accuracy by Feature Column')
ax.set_xlim(0.6, 0.95)
max_acc = classification_sorted['Accuracy'].max()
for i, (feature, acc) in enumerate(zip(classification_sorted['Feature Column'], classification_sorted['Accuracy'])):
    if acc == max_acc:
        bars[i].set_color('#ea97ad')
        ax.text(acc + 0.01, i, f'★ Best: {acc:.4f}', va='center')
    else:
        ax.text(acc + 0.005, i, f'{acc:.4f}', va='center')
plt.tight_layout()
plt.savefig('netflix_fig9_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 9")

# Chart 10
countries = ['United States', 'India', 'United Kingdom', 'Japan', 'South Korea', 'Canada']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
for idx, country in enumerate(countries):
    if country in actors_files:
        try:
            actors_df = pd.read_csv(actors_files[country])
            top10 = actors_df.head(10).sort_values('count', ascending=True)
            ax = axes[idx]
            ax.barh(top10['actor'], top10['count'], color='#ea97ad')
            ax.set_title(f'Top 10 Actors - {country}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Appearances', fontsize=10)
            for i, (actor, count) in enumerate(zip(top10['actor'], top10['count'])):
                ax.text(count + 0.3, i, str(int(count)), va='center', fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
        except:
            axes[idx].text(0.5, 0.5, f'Data not available', ha='center', va='center')
            axes[idx].set_title(f'Top Actors - {country}')
    else:
        axes[idx].text(0.5, 0.5, f'No data', ha='center', va='center')
        axes[idx].set_title(f'Top Actors - {country}')
plt.suptitle('Most Popular Actors by Country (Netflix Data)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('netflix_fig10_actors.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Chart 10")

print("=" * 60)
print("✅ ALL 10 NETFLIX CHARTS GENERATED SUCCESSFULLY!")
print("=" * 60)