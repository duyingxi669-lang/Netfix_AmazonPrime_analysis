import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from collections import Counter
from pyspark.sql.types import IntegerType, FloatType

# ============================================================
# 3.1 查看数据集基本信息
# ============================================================

# 读取CSV文件
titles = pd.read_csv('D:/DPWII project/Amazon Prime dataset/titles.csv')       # 影视作品主表
credits = pd.read_csv('D:/DPWII project/Amazon Prime dataset/credits.csv')        # 人物关系表

# 显示前几行数据
print("========== (titles.csv) First few rows of the data ==========")
print(titles.head())
print("========== (credits.csv) First few rows of the data ==========")
print(credits.head())

# 数据集大小
print("\n========== The size of the dataset ==========")
print("The size of titles: ",titles.shape)
print("The size of credits: ",credits.shape)

# 显示数据集的概要信息
print("\n========== (titles.csv) Data information ==========")
print(titles.info())
print("\n========== (credits.csv) Data information ==========")
print(credits.info())

# ============================================================
# 3.2.1 检查和处理作品表空值
# ============================================================
# 检查数据集中每列的缺失值情况
print("\n========== (titles.csv) Missing rate for each column ==========")
for i in titles.columns:
    null_rate = titles[i].isna().sum() / len(titles) * 100
    if null_rate > 0:
        print("{} null rate: {}%".format(i, round(null_rate, 2)))

# 将age_certification列中的空值用该列中出现最频繁的值进行替换
titles['age_certification'] = titles['age_certification'].fillna(titles['age_certification'].mode()[0])

# genres：用 'No Data' 填充
titles['genres'] = titles['genres'].fillna('No Data')
# description：用 'No Data' 填充
titles['description'] = titles['description'].fillna('No Data')
# imdb_score,imdb_votes, tmdb_score, tmdb_popularity：用中位数填充
titles['imdb_score'] = titles['imdb_score'].fillna(titles['imdb_score'].median())
titles['tmdb_score'] = titles['tmdb_score'].fillna(titles['tmdb_score'].median())
titles['tmdb_popularity'] = titles['tmdb_popularity'].fillna(titles['tmdb_popularity'].median())
titles['imdb_votes'] = titles['imdb_votes'].fillna(titles['imdb_votes'].median())

# 丢弃其余含有空值的行（title、type 等关键字段不能为空）
titles.dropna(subset=['title', 'type', 'release_year'], inplace=True)

# 丢弃数据集中的重复行
titles.drop_duplicates(subset=['id'], inplace=True)
titles.drop_duplicates(inplace=True)

# 统计数据集中每列的缺失值数量
print("\n========== (titles.csv) The number of missing values for each column after processing ==========")
print(titles.isnull().sum())

# ============================================================
# 3.2.1 检查和处理人物表空值
# ============================================================
print("\n========== (credits.csv) Missing rate for each column ==========")
for i in credits.columns:
    null_rate = credits[i].isna().sum() / len(credits) * 100
    if null_rate > 0:
        print("{} null rate: {}%".format(i, round(null_rate, 2)))

# 将空值用字符串'No Data'进行替换
credits['name'] = credits['name'].fillna('No Data')
credits['character'] = credits['character'].fillna('No Data')

# 丢弃重复行
credits.drop_duplicates(inplace=True)

print("\n========== (credits.csv) The number of missing values for each column after processing ==========")
print(credits.isnull().sum())

# ============================================================
# 3.3 增加新字段
# ============================================================
# 增加特征值
# 提取主要题材
titles['main_genre'] = titles['genres'].apply(
    lambda x: x.strip("[]'").split(",")[0].strip().strip("'") 
    if isinstance(x, str) and x != 'No Data' else 'No Data'
)

# 处理列表格式（去除方括号和引号）
titles['production_countries'] = titles['production_countries'].astype(str)
titles['production_countries'] = titles['production_countries'].str.replace("[", "").str.replace("]", "").str.replace("'", "")

# 清理 genres 列表格式（保留完整题材信息）
titles['genres'] = titles['genres'].astype(str)
titles['genres'] = titles['genres'].str.replace("[", "", regex=False)
titles['genres'] = titles['genres'].str.replace("]", "", regex=False)
titles['genres'] = titles['genres'].str.replace("'", "", regex=False)

# 提取年代（Amazon数据集年份跨度大）
titles['decade'] = (titles['release_year'] // 10 * 10).astype(int)

# 计算演员数量
actor_count = credits[credits['role'] == 'ACTOR'].groupby('id').size().reset_index(name='actor_count')
titles = titles.merge(actor_count, on='id', how='left')
titles['actor_count'] = titles['actor_count'].fillna(0).astype(int)

# 提取每部作品的演员名单（用逗号拼接）
actors_by_id = credits[credits['role'] == 'ACTOR'] \
    .groupby('id')['name'].apply(lambda x: ','.join(x)).reset_index(name='cast')
titles = titles.merge(actors_by_id, on='id', how='left')
titles['cast'] = titles['cast'].fillna('No Data')

# 查看添加新列后的数据
print("\n========== The first 5 rows of data after adding new columns ==========")
print(titles.head())

# 保存处理后的文件
titles.to_csv('D:/DPWII project/Amazon Prime dataset/amazon_cleaned.csv', index=False)
print("\n========== The data is preprocessed and the file is saved ==========")

# ============================================================
# 4. 数据分析（Amazon Prime）
# ============================================================

# 创建 SparkSession 对象
spark = SparkSession.builder.appName("AmazonAnalysis").getOrCreate()

# ============================================================
# 4.1 读取数据
# ============================================================
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("quote", '"') \
    .option("escape", '"') \
    .option("multiLine", "true") \
    .csv('D:/DPWII project/Amazon Prime dataset/amazon_cleaned.csv')

# 转换数值列类型
df = df.withColumn("release_year", F.col("release_year").cast(IntegerType()))
df = df.withColumn("runtime", F.col("runtime").cast(IntegerType()))
df = df.withColumn("actor_count", F.col("actor_count").cast(IntegerType()))
df = df.withColumn("decade", F.col("decade").cast(IntegerType()))
df = df.withColumn("imdb_score", F.col("imdb_score").cast(FloatType()))
df = df.withColumn("imdb_votes", F.col("imdb_votes").cast(FloatType()))
df = df.withColumn("tmdb_score", F.col("tmdb_score").cast(FloatType()))
df = df.withColumn("tmdb_popularity", F.col("tmdb_popularity").cast(FloatType()))

print("\n========== 4.1 Load the data ==========")
# 检查读出的数据是否无误
print("The number of rows: ", df.count())
print("The number of columns: ", len(df.columns))
df.printSchema()
df.show(5)

# ============================================================
# 4.2 分析电影和电视剧的占比情况
# ============================================================
print("\n========== 4.2 Proportion of movies and TV shows ==========")

def movies_tvshows(df):
    col = "type"
    df_movies_tvshows = df.groupBy(col).count().withColumnRenamed("count", "count")
    return df_movies_tvshows

# 调用movies_tvshows函数并获取结果
df_movies_tvshows = movies_tvshows(df)
# 显示数据分析结果
df_movies_tvshows.show()
# 将数据分析结果保存为CSV文件
df_movies_tvshows.toPandas().to_csv("output/amazon_movies_tvshows.csv",index=False)

# ============================================================
# 4.3 按发布年份统计电影和电视剧数量
# ============================================================
print("\n========== 4.3 Count the number of movies and TV shows by released year ==========")

def release_overyear(df):
    tv_show_df = df.filter(df["type"] == "SHOW")
    movie_df = df.filter(df["type"] == "MOVIE")
    
    # 电视剧
    tv_show_count = tv_show_df.groupBy("release_year").count()
    tv_show_count = tv_show_count.withColumn(
        "percent", (F.col("count") / tv_show_df.count()) * 100
    ).orderBy("release_year")
    
    # 电影
    movie_count = movie_df.groupBy("release_year").count()
    movie_count = movie_count.withColumn(
        "percent", (F.col("count") / movie_df.count()) * 100
    ).orderBy("release_year")
    
    return tv_show_count, movie_count

released_tv_show, released_movie = release_overyear(df)

# 清除null值
released_tv_show = released_tv_show.dropna()
released_movie = released_movie.dropna()

print("Count the number of TV shows by released year: ")
released_tv_show.show(10)
print("Count the number of movies by released year: ")
released_movie.show(10)

# 保存结果
released_tv_show.toPandas().to_csv('output/amazon_released_tv_show.csv', index=False)
released_movie.toPandas().to_csv('output/amazon_released_movie.csv', index=False)

# ============================================================
# 4.4 统计不同国家的影视剧总量
# ============================================================
print("\n========== 4.4 The total number of movies and TV shows in different countries ==========")

def country_count(df):
    # 拆分 production_countries（多个国家用逗号分隔），展开成多行
    df_exploded = df.select(F.explode(F.split(F.col("production_countries"), ",")).alias("country"))
    
    # 去除空格
    df_exploded = df_exploded.withColumn("country", F.trim(F.col("country")))
    
    # 去除空值
    df_clean = df_exploded.filter(F.col("country") != "").na.drop()
    
    # 分组计数并转为字典
    country_dict = df_clean.groupBy("country").count().rdd.collectAsMap()
    return country_dict

country = country_count(df)

# 查看前10名
sorted_country = sorted(country.items(), key=lambda x: x[1], reverse=True)
for k, v in sorted_country[:10]:
    print(f"{k}: {v}")

# 保存为CSV
df_country = pd.DataFrame.from_dict(country, orient='index', columns=['count'])
df_country = df_country.sort_values('count', ascending=False)
df_country.to_csv('output/amazon_country_data.csv', index_label='country')

# ============================================================
# 4.5 分析不同国家的电影和电视剧占比
# ============================================================
print("\n========== 4.5 Proportion of movies and TV shows in different countries ==========")

# 先拆分 production_countries，展开成每个国家一行
df_exploded = df.withColumn("country", F.explode(F.split(F.col("production_countries"), ",")))
df_exploded = df_exploded.withColumn("country", F.trim(F.col("country")))
df_exploded = df_exploded.filter(F.col("country") != "")

def compute_ratio(df):
    # 统计每个国家的出现次数
    df_country_counts = df.groupBy('country').count()

    # 获取出现次数最多的前10个国家
    top_countries = df_country_counts.orderBy(F.col('count').desc()).limit(10).select('country')

    # 内连接
    df_top_countries = df.join(top_countries, on='country', how='inner')

    # 统计每个国家中各类型（电影/电视剧）的出现次数
    df_type_counts = df_top_countries.groupBy('country', 'type').count()

    # 窗口函数计算总次数
    window = Window.partitionBy('country')
    df_total_counts = df_type_counts.withColumn('total', F.sum('count').over(window))

    # 计算比例
    df_ratio = df_total_counts.withColumn('ratio', F.col('count') / F.col('total'))

    # 筛选
    df_ratio = df_ratio.filter((df_ratio.type == 'MOVIE') | (df_ratio.type == 'SHOW'))

    # 使用透视表将数据按国家进行汇总，缺失值用0填充
    pivot_df = df_ratio.groupBy('country').pivot('type').sum('ratio').fillna(0)

    return pivot_df

data_q2q3 = compute_ratio(df_exploded)
data_q2q3.show()

# 保存
data_q2q3.toPandas().to_csv('output/amazon_data_q2q3.csv', index=False)

# ============================================================
# 4.6 分析电影时长的分布
# ============================================================
print("\n========== 4.6 Distribution of movie lengths ==========")

def movie_duration(df):
    # 过滤出电影，提取时长
    result_df = df.filter(df.type == 'MOVIE') \
        .select(F.col("runtime").cast("float").alias("duration")) \
        .filter(F.col("duration").isNotNull())
    return result_df

df_movie_duration = movie_duration(df)
df_movie_duration.show(10)

# 保存
df_movie_duration.toPandas().to_csv('output/amazon_movie_duration.csv', index=False)

# ============================================================
# 4.7 分析电视剧和电影的评分分布
# ============================================================
print("\n========== 4.7 Distribution of ratings for movies and TV shows ==========")

def rating_content(df):
    # TV Show 评分统计（按评分区间分组）
    vc1 = df.filter(F.col("type") == "SHOW") \
        .withColumn("score_range",
            F.when(F.col("imdb_score") >= 9, "9-10")
             .when(F.col("imdb_score") >= 8, "8-9")
             .when(F.col("imdb_score") >= 7, "7-8")
             .when(F.col("imdb_score") >= 6, "6-7")
             .when(F.col("imdb_score") >= 5, "5-6")
             .otherwise("<5")
        ) \
        .groupBy("score_range") \
        .count() \
        .orderBy("score_range")
    
    vc1 = vc1.withColumnRenamed("score_range", "index") \
        .withColumnRenamed("count", "count_tvshow")
    
    total_count_tvshow = vc1.agg(F.sum("count_tvshow")).collect()[0][0]
    vc1 = vc1.withColumn("percent_tvshow", F.col("count_tvshow") * 100 / total_count_tvshow)

    # Movie 评分统计（按评分区间分组）
    vc2 = df.filter(F.col("type") == "MOVIE") \
        .withColumn("score_range",
            F.when(F.col("imdb_score") >= 9, "9-10")
             .when(F.col("imdb_score") >= 8, "8-9")
             .when(F.col("imdb_score") >= 7, "7-8")
             .when(F.col("imdb_score") >= 6, "6-7")
             .when(F.col("imdb_score") >= 5, "5-6")
             .otherwise("<5")
        ) \
        .groupBy("score_range") \
        .count() \
        .orderBy("score_range")
    
    vc2 = vc2.withColumnRenamed("score_range", "index") \
        .withColumnRenamed("count", "count_movie")
    
    total_count_movie = vc2.agg(F.sum("count_movie")).collect()[0][0]
    vc2 = vc2.withColumn("percent_movie", F.col("count_movie") * 100 / total_count_movie)
    
    return vc1, vc2

df_rating_tvshow, df_rating_movie = rating_content(df)

print("Distribution of ratings for TV shows：")
df_rating_tvshow.show(20)
print("Distribution of ratings for movies：")
df_rating_movie.show(20)

# 保存
df_rating_tvshow.toPandas().to_csv('output/amazon_rating_tvshow.csv', index=False)
df_rating_movie.toPandas().to_csv('output/amazon_rating_movie.csv', index=False)

# ============================================================
# 4.8 分析电影和电视剧的题材分布
# ============================================================
print("\n========== 4.8 Distribution of genres in movies and TV shows ==========")

def categories_content(df):
    # 提取 genre 列，去除空值
    df_exploded = df.select(F.col("genres")).na.drop() \
        .filter(F.col("genres") != "No Data") \
        .filter(F.col("genres") != "") \
        .select(F.explode(F.split("genres", ",")).alias("category"))
    df_cleaned = df_exploded.withColumn("category", F.trim(F.col("category")))
    
    # 转为 Pandas，用 Counter 统计
    categories = df_cleaned.toPandas()["category"].tolist()
    counter_list = Counter(categories).most_common(50)  # 取前50
    
    labels = [x[0] for x in counter_list][::-1]
    values = [x[1] for x in counter_list][::-1]
    
    categories_pd = pd.DataFrame({"Category": labels, "Count": values})
    return categories_pd

categories_df = categories_content(df)
print(categories_df.head(20))

# 保存
categories_df.to_csv('output/amazon_categories.csv', index=False)

# ============================================================
# 4.9 分析前10个国家最受欢迎的电影明星
# ============================================================
print("\n========== 4.9 The most popular movie star in top 10 countries ==========")
# ============================================================
def top_actors(df, country_code):
    df_with_country = df.withColumn(
        "from_country", 
        (F.lower(F.col("production_countries")).like("%" + country_code.lower() + "%")).cast("int") 
    )

    country_movies = df_with_country.filter(
        (F.col("from_country") == 1) & 
        (F.col("runtime").isNotNull()) & 
        (F.col("cast") != "No Data")
    )

    cast = country_movies.select(F.explode(F.split(F.col("cast"), ",")).alias("actor")) \
        .withColumn("actor", F.trim(F.col("actor"))) \
        .groupBy("actor").count().orderBy(F.col("count").desc())

    cast = cast.filter(F.col("actor") != "")

    cast_pandas = cast.limit(25).toPandas()
    # 预览前 5 名演员
    print(f"\n{country_code} Top 5 most popular actors: ")
    print(cast_pandas.head(5).to_string(index=False))
    cast_pandas.to_csv(f"output/amazon_actors_{country_code}.csv", index=False)

# 前10个国家
top_actors_titles = ["US", "IN", "GB", "CA", "FR", "JP", "DE", "CN", "AU", "IT"]

for country_code in top_actors_titles:
    top_actors(df, country_code)

print("\nTop 10 country actor data saved")

# ============================================================
# 4.10 分析不同特征列对影视剧分类的效果
# ============================================================
print("\n========== 4.10 The effect of different features on the classification of movies and TV shows ==========")

def classification(df, feature_columns):
    # 创建 label 索引
    label_indexer = StringIndexer(inputCol="type", outputCol="label")
    
    pipelines = []
    accuracies = []

    for column in feature_columns:
        if df.select(column).distinct().count() > 2:
            string_indexer = StringIndexer(inputCol=column, outputCol=column + "_index", handleInvalid="keep")
            onehot_encoder = OneHotEncoder(inputCols=[column + "_index"], outputCols=[column + "_encoded"])
            assembler = VectorAssembler(inputCols=[column + "_encoded"], outputCol="features")
            classifier = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)
            pipeline = Pipeline(stages=[label_indexer, string_indexer, onehot_encoder, assembler, classifier])
        else:
            string_indexer = StringIndexer(inputCol=column, outputCol=column + "_index", handleInvalid="keep")
            assembler = VectorAssembler(inputCols=[column + "_index"], outputCol="features")
            classifier = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10)
            pipeline = Pipeline(stages=[label_indexer, string_indexer, assembler, classifier])
        
        pipelines.append(pipeline)

    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

    for i, pipeline in enumerate(pipelines):
        model = pipeline.fit(train_data)
        predictions = model.transform(test_data)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        accuracies.append(accuracy)
        print(f"Accuracy using '{feature_columns[i]}': {accuracy:.4f}")

    max_accuracy = max(accuracies)
    best_feature = feature_columns[accuracies.index(max_accuracy)]
    print(f"\nBest feature column: {best_feature}")
    print(f"Highest accuracy: {max_accuracy:.4f}")

    return accuracies, feature_columns

# 排除不需要的列
feature_columns = [col for col in df.columns if col not in ("type", "id", "imdb_id","seasons")]

accuracies, feature_columns = classification(df, feature_columns)

# 保存
classification_df = pd.DataFrame({'Feature Column': feature_columns, 'Accuracy': accuracies})
classification_df.to_csv('output/amazon_classification.csv', index=False)

# Extra analysis
# ============================================================
# 4.11：统计不同国家的平均评分
# ============================================================
print("\n========== 4.11 Average score in different countries ==========")

def country_avg_score(df, df_exploded):
    country_score = df_exploded.groupBy("country") \
        .agg(F.avg("imdb_score").alias("avg_score"), F.count("*").alias("count")) \
        .filter(F.col("count") > 20) \
        .orderBy(F.desc("avg_score"))
    return country_score

country_score = country_avg_score(df, df_exploded)
country_score.show(20, truncate=False)
country_score.toPandas().to_csv("output/amazon_country_avg_score.csv", index=False)


# ============================================================
# 4.12：统计不同年代的平均评分
# ============================================================
print("\n========== 4.12：Average score in different decades ==========")

def decade_avg_score(df):
    decade_score = df.filter(F.col("imdb_score").isNotNull()) \
        .groupBy("decade") \
        .agg(F.avg("imdb_score").alias("avg_score"), F.count("*").alias("count")) \
        .filter(F.col("count") > 10) \
        .orderBy("decade")
    return decade_score

decade_score = decade_avg_score(df)
decade_score.show(30, truncate=False)
decade_score.toPandas().to_csv("output/amazon_decade_avg_score.csv", index=False)


# ============================================================
# 4.13：电影和电视剧的评分分析
# ============================================================
print("\n========== 4.13 The imdb_score for movies and TV shows ==========")

def type_score_compare(df):
    type_score = df.filter(F.col("imdb_score").isNotNull()) \
        .groupBy("type") \
        .agg(F.avg("imdb_score").alias("avg_score"),
             F.stddev("imdb_score").alias("std_score"),
             F.min("imdb_score").alias("min_score"),
             F.max("imdb_score").alias("max_score"),
             F.count("*").alias("count"))
    return type_score

type_score = type_score_compare(df)
type_score.show(truncate=False)
type_score.toPandas().to_csv("output/amazon_type_score_compare.csv", index=False)


# ============================================================
# 4.14：不同题材的平均评分
# ============================================================
print("\n========== 4.14 Average score in different genres ==========")

def genre_avg_score(df):
    df_genre = df.select(F.explode(F.split("genres", ",")).alias("genre"), "imdb_score")
    df_genre = df_genre.withColumn("genre", F.trim(F.col("genre")))
    df_genre = df_genre.filter(F.col("genre") != "").filter(F.col("genre") != "No Data")
    df_genre = df_genre.filter(F.col("imdb_score").isNotNull())
    
    genre_score = df_genre.groupBy("genre") \
        .agg(F.avg("imdb_score").alias("avg_score"), F.count("*").alias("count")) \
        .filter(F.col("count") > 20) \
        .orderBy(F.desc("avg_score"))
    return genre_score

genre_score = genre_avg_score(df)
genre_score.show(30, truncate=False)
genre_score.toPandas().to_csv("output/amazon_genre_avg_score.csv", index=False)