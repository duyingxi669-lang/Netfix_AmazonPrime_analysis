import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from collections import Counter

# ============================================================
# 3.1 查看数据集基本信息
# ============================================================

# 读取CSV文件
netflix = pd.read_csv('D:/DPWII project/netflix_titles.csv', encoding='latin-1').iloc[:, :12]

# 显示前几行数据
print("========== The first few lines of data ==========")
print(netflix.head())

# 数据集大小
print("\n========== The size of datasets ==========")
print(netflix.shape)

# 显示数据集的概要信息
print("\n========== Dataset information ==========")
print(netflix.info())


# ============================================================
# 3.2 检查和处理空值
# ============================================================

# 检查数据集中每列的缺失值情况
print("\n========== The missing rate of each column ==========")
for i in netflix.columns:
    null_rate = netflix[i].isna().sum() / len(netflix) * 100
    if null_rate > 0:
        print("{} null rate: {}%".format(i, round(null_rate, 2)))

# 将country列中的空值用该列中出现最频繁的值进行替换
netflix['country'] = netflix['country'].fillna(netflix['country'].mode()[0])

# 将空值用字符串'No Data'进行替换
netflix['cast'] = netflix['cast'].replace(np.nan, 'No Data')
netflix['director'] = netflix['director'].replace(np.nan, 'No Data')

# 丢弃数据集中含有空值的行
netflix.dropna(inplace=True)

# 丢弃数据集中的重复行
netflix.drop_duplicates(inplace=True)

# 统计数据集中每列的缺失值数量
print("\n========== The number of missing values for each column after processing ==========")
print(netflix.isnull().sum())


# ============================================================
# 3.3 增加新字段
# ============================================================
# 增加特征值
# 转换为日期时间格式
netflix["date_added"] = pd.to_datetime(netflix['date_added'])

# 提取出年份信息，并将结果存储在新的列中
netflix['year_added'] = netflix['date_added'].dt.year

# 查看添加新列后的数据
print("\n==========  The first 5 rows of data after adding new columns ==========")
print(netflix.head())

# 保存处理后的文件
netflix.to_csv('D:/DPWII project/netflix_data.csv', index=False)
print("\n========== The data is preprocessed and the file is saved ==========")

# ============================================================
# 4. 数据分析
# ============================================================

# 创建SparkSession对象
spark = SparkSession.builder.appName("NetflixAnalysis").getOrCreate()

# ============================================================
# 4.1 读取数据
# ============================================================
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("escape", '"') \
    .csv('D:/DPWII project/netflix_data.csv')

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
    # 指定要进行分组的列名为“type”
    col="type"
    # 对输入的DataFrame进行分组和技术，将计数列重命名为“count”
    df_movies_tvshows = df.groupBy(col).count().withColumnRenamed("count","count")
    # 返回数据分析结果的DataFrame
    return df_movies_tvshows
# 调用movies_tvshows函数并获取结果
df_movies_tvshows = movies_tvshows(df)
# 显示数据分析结果
df_movies_tvshows.show()
# 将数据分析结果保存为CSV文件
df_movies_tvshows.toPandas().to_csv("D:/DPWII project/Netflix_output/movies_tvshows.csv",index=False)

# ============================================================
# 4.3 按年份统计添加的电视节目和电影数量
# ============================================================
print("\n========== 4.3 Statistical analysis of the number of TV shows and movies added by year ==========")
# 定义函数added_overyear，用于执行按年份统计数据添加的电视节目和电影数量
def added_overyear(df):
    tv_show_df = df.filter(df["type"] == "TV Show")
    movie_df = df.filter(df["type"] == "Movie")
    # 电视节目
    tv_show_count = tv_show_df.groupBy("year_added").count()
    tv_show_count = tv_show_count.withColumn(
        "percent", (F.col("count") / tv_show_df.count()) * 100
    ).orderBy("year_added")
    # 电影
    movie_count = movie_df.groupBy("year_added").count()
    movie_count = movie_count.withColumn(
        "percent", (F.col("count") / movie_df.count()) * 100
    ).orderBy("year_added")
    
    return tv_show_count, movie_count

added_tv_show, added_movie = added_overyear(df)
added_tv_show.show(),added_movie.show()

# 清除null值
added_tv_show = added_tv_show.dropna()
added_movie = added_movie.dropna()

added_tv_show.show()
added_movie.show()

# 将结果保存为csv文件
added_tv_show.toPandas().to_csv('D:/DPWII project/Netflix_output/added_tv_show.csv', index=False)
added_movie.toPandas().to_csv('D:/DPWII project/Netflix_output/added_movie.csv', index=False)

# ============================================================
# 4.4 按年份统计发布的电视节目和电影数量
# ============================================================
print("\n========== 4.4 Number of TV shows and movies released by year ==========")

def release_overyear(df):
    tv_show_df = df.filter(df["type"] == "TV Show")
    movie_df = df.filter(df["type"] == "Movie")
    
    tv_show_count = tv_show_df.groupBy("release_year").count()
    tv_show_count = tv_show_count.withColumn(
        "percent", (F.col("count") / tv_show_df.count()) * 100
    ).orderBy("release_year")
    
    movie_count = movie_df.groupBy("release_year").count()
    movie_count = movie_count.withColumn(
        "percent", (F.col("count") / movie_df.count()) * 100
    ).orderBy("release_year")
    
    return tv_show_count, movie_count

released_tv_show, released_movie = release_overyear(df)

# 清除null值
released_tv_show = released_tv_show.dropna()
released_movie = released_movie.dropna()

released_tv_show.show()
released_movie.show()
# 将结果保存为csv文件
released_tv_show.toPandas().to_csv('D:/DPWII project/Netflix_output/released_tv_show.csv', index=False)
released_movie.toPandas().to_csv('D:/DPWII project/Netflix_output/released_movie.csv', index=False)

# ============================================================
# 4.5 统计不同国家的影视剧总量
# ============================================================
print("\n========== 4.5 The total number of movies and tv shows in different countries ==========")
def country_count(df):
    df = df.select("country").na.drop()
    country_dict = df.groupBy("country").count().alias("count")
    country_dict = country_dict.rdd.collectAsMap()
    return country_dict

country = country_count(df)
df_country = pd.DataFrame.from_dict(country, orient='index', columns=['count'])
print("Total number of movies and TV shows in the top 10 countries: ")
print(df_country.sort_values('count', ascending=False).head(10))
df_country = pd.DataFrame.from_dict(country, orient='index', columns=['count'])
df_country.to_csv('D:/DPWII project/Netflix_output/country_data.csv',index_label='country')

# ============================================================
# 4.6 分析不同国家的电影和电视剧占比
# ============================================================
print("\n========== 4.6 The proportion of movies and tv shows in different countries. ==========")
def compute_ratio(df):
    # 统计每个国家的出现次数
    df_country_counts = df.groupBy('country').count()

    # 获取出现次数最多的前10个国家
    top_countries = df_country_counts.orderBy(F.col('count').desc()).limit(10).select('country')

    # 将原始数据与出现次数最多的国家进行内连接
    df_top_countries = df.join(top_countries, on='country', how='inner')

    # 统计每个国家中各类型（电影/电视剧）的出现次数
    df_type_counts = df_top_countries.groupBy('country', 'type').count()

    # 使用窗口函数计算每个国家的总出现次数
    window = Window.partitionBy('country')
    df_total_counts = df_type_counts.withColumn('total', F.sum('count').over(window))

    # 计算每个国家中各类型的比例
    df_ratio = df_total_counts.withColumn('ratio', F.col('count') / F.col('total'))

    # 筛选出电影和电视剧类型的数据
    df_ratio = df_ratio.filter((df_ratio.type == 'Movie') | (df_ratio.type == 'TV Show'))

    # 使用透视表将数据按国家进行汇总，缺失值用0填充
    pivot_df = df_ratio.groupBy('country').pivot('type').sum('ratio').fillna(0)

    return pivot_df

# 调用函数
data_q2q3 = compute_ratio(df)
data_q2q3.show()

# 保存结果
data_q2q3.toPandas().to_csv('D:/DPWII project/Netflix_output/data_q2q3.csv', index=False)

# ============================================================
# 4.7 分析电影时长的分布
# ============================================================
print("\n========== 4.7 Distribution of movie lengths ==========")
def movie_duration(df):
    # 过滤出类型为电影的行，提取时长数字，转换为浮点数
    result_df = df.filter(df.type == 'Movie').select(
        F.regexp_extract(df.duration, r'(\d+)', 1).cast('float').alias('duration')
    )
    return result_df

df_movie_duration = movie_duration(df)
df_movie_duration.show(10)

# 保存
df_movie_duration.toPandas().to_csv('D:/DPWII project/Netflix_output/movie_duration.csv', index=False)

# ============================================================
# 4.8 分析电视剧和电影的评分分布
# ============================================================
print("\n========== 4.8 Distribution of ratings for movies and TV shows ==========")
def rating_content(df):
    # TV Show 评分分布
    vc1 = df.filter(F.col("type") == "TV Show") \
        .groupBy(F.col("rating")) \
        .count() \
        .orderBy(F.col("rating"))
    vc1 = vc1.withColumnRenamed("rating", "index") \
        .withColumnRenamed("count", "count_tvshow")
    
    total_count_tvshow = vc1.agg(F.sum("count_tvshow")).collect()[0][0]
    vc1 = vc1.withColumn("percent_tvshow", F.col("count_tvshow") * 100 / total_count_tvshow)

    # Movie 评分分布
    vc2 = df.filter(F.col("type") == "Movie") \
        .groupBy(F.col("rating")) \
        .count() \
        .orderBy(F.col("rating"))
    vc2 = vc2.withColumnRenamed("rating", "index") \
        .withColumnRenamed("count", "count_movie")
    
    total_count_movie = vc2.agg(F.sum("count_movie")).collect()[0][0]
    vc2 = vc2.withColumn("percent_movie", F.col("count_movie") * 100 / total_count_movie)
    
    return vc1, vc2
df_rating_tvshow, df_rating_movie = rating_content(df)
df_rating_tvshow = df_rating_tvshow.na.drop()
df_rating_movie = df_rating_movie.na.drop()

print("Distribution of ratings for TV shows: ")
df_rating_tvshow.show(20)
print("Distribution of ratings for movies: ")
df_rating_movie.show(20)

# 保存
df_rating_tvshow.toPandas().to_csv('D:/DPWII project/Netflix_output/rating_tvshow.csv', index=False)
df_rating_movie.toPandas().to_csv('D:/DPWII project/Netflix_output/rating_movie.csv', index=False)

# ============================================================
# 4.9 分析电影和电视剧的题材分布
# ============================================================
print("\n========== 4.9 Distribution of genres in movies and TV shows ==========")
def categories_content(df):
    # 提取所有的listed_in列数据（修改：用 split + explode 替代 rdd.flatMap）
    df_exploded = df.select(F.col("listed_in")).na.drop() \
        .select(F.explode(F.split("listed_in", ",")).alias("category"))
    df_cleaned = df_exploded.withColumn("category", F.trim(F.col("category")))
    categories = df_cleaned.toPandas()["category"].tolist()
    
    categories = [item.strip() for sublist in categories if sublist is not None 
                  for item in (sublist.split(",") if sublist else [])]
    
    counter_list = Counter(categories).most_common(50)
    
    labels = [_[0] for _ in counter_list][::-1]
    values = [_[1] for _ in counter_list][::-1]
    
    categories_pd = pd.DataFrame({"Category": labels, "Count": values})
    return categories_pd
categories_df=categories_content(df)
# 调用函数获取分类统计结果
print(categories_df.head(20))
# 保存
categories_df.to_csv('D:/DPWII project/Netflix_output/categories.csv', index=False)

# ============================================================
# 4.10 分析6个国家最受欢迎的电影明星
# ============================================================
print("\n========== 4.10 The most popular movie star in 6 countries ==========")
def top_actors(df, country_name):
    # 添加 from_country 列，判断电影是否来自所选国家
    df_with_country = df.withColumn(
        "from_country", 
        (F.lower(F.col("country")).like("%" + country_name.lower() + "%")).cast("int") 
    )

    # 筛选出来自所选国家的电影
    country_movies = df_with_country.filter(
        (F.col("from_country") == 1) & 
        (F.col("duration") != "") & 
        (F.col("cast") != "No Data")
    )

    # 提取演员信息并展开，按演员分组统计，按频次降序排序
    cast = country_movies.select(F.explode(F.split(F.col("cast"), ",")).alias("actor")) \
        .groupBy("actor").count().orderBy(F.col("count").desc())

    # 过滤掉演员为空的行
    cast = cast.filter(F.col("actor") != "")

    # 将结果转换为 Pandas DataFrame，限制结果为前25个演员
    cast_pandas = cast.limit(25).toPandas()
    # 预览结果
    print(f"\n{country_name} Top 5 most popular actors: ")
    print(cast_pandas.head(5).to_string(index=False))

    # 保存中间结果 CSV
    safe_name = country_name.replace(" ", "_")
    cast_pandas.to_csv(f"D:/DPWII project/Netflix_output/actors_{safe_name}.csv", index=False)

    # 提取演员和频次
    #labels = cast_pandas["actor"]
    #values = cast_pandas["count"]

    # 创建水平柱状图
    #actor = go.Bar(y=labels[::-1], x=values[::-1], orientation="h",name="", marker=dict(color="#ea97ad"))

    #return actor

# 定义要分析的国家列表
top_actors_titles = ["United States", "India", "United Kingdom", "Japan", "South Korea", "Canada"]

actors = []
for i in range(len(top_actors_titles)):
    country_title = top_actors_titles[i]
    if country_title != "":
        actors.append(top_actors(df, country_title))

# 查看前3个国家的图表数据
#for actor in actors[:3]:
    #actor.show()

# 保存结果
#actors_data_df = pd.DataFrame(actors)
#actors_data_df.to_csv("D:/DPWII project/actors_data.csv", index=False)

# ============================================================
# 4.11 分析不同特征列对影视剧分类的效果
# ============================================================
print("\n========== 4.11 The effect of different features on the classification of movies and TV shows ==========")
def classification(df, feature_columns):
    # 定义存储不同特征的Pipeline和准确率列表
    pipelines = []
    accuracies = []

    for column in feature_columns:
        # 创建 label 索引
        label_indexer = StringIndexer(inputCol="type", outputCol="label")

        # 判断是否需要 OneHotEncoder
        if df.select(column).distinct().count() > 2:
            # 特征值 > 2：使用 OneHotEncoder
            string_indexer = StringIndexer(inputCol=column, outputCol=column + "_index")
            string_indexer.setHandleInvalid("keep")
            onehot_encoder = OneHotEncoder(inputCols=[string_indexer.getOutputCol()], outputCols=[column + "_encoded"])
            assembler = VectorAssembler(inputCols=[onehot_encoder.getOutputCols()[0]], outputCol="features")
        else:
            # 特征值 <= 2：不需要 OneHotEncoder
            string_indexer = StringIndexer(inputCol=column, outputCol=column + "_index")
            string_indexer.setHandleInvalid("keep")
            assembler = VectorAssembler(inputCols=[string_indexer.getOutputCol()], outputCol="features")
        
        # 创建RandomForestClassifier作为分类器
        classifier = RandomForestClassifier(featuresCol="features", labelCol="label")

        # 根据特征唯一值梳理选择Pipeline中的阶段
        if df.select(column).distinct().count() > 2:
            pipeline = Pipeline(stages=[label_indexer, string_indexer, onehot_encoder, assembler, classifier])
        else:
            pipeline = Pipeline(stages=[label_indexer, string_indexer, assembler, classifier])
        pipelines.append(pipeline)

    # 拆分训练集和测试集
    train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

    for i, pipeline in enumerate(pipelines):
        # 训练模型
        model = pipeline.fit(train_data)
        # 对测试集预测
        predictions = model.transform(test_data)
        # 计算准确率
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        accuracies.append(accuracy)
        
        print(f"Accuracy using '{feature_columns[i]}' as the feature column: {accuracy}")

    # 找到最高准确率及其对应的特征列
    max_accuracy = max(accuracies)
    best_feature_column = feature_columns[accuracies.index(max_accuracy)]
    print(f"Best feature column: {best_feature_column}")
    print(f"Highest accuracy: {max_accuracy}")

    return accuracies, feature_columns

# 获取除了 type 和 show_id 和 date_added (year_added可以代表) 以外的所有列作为特征列
feature_columns = [col for col in df.columns if col not in ("type", "show_id", "date_added")]

# 调用函数
accuracies, feature_columns = classification(df, feature_columns)

# 保存结果
classification_df = pd.DataFrame({'Feature Column': feature_columns, 'Accuracy': accuracies})
classification_df.to_csv('D:/DPWII project/Netflix_output/classification.csv', index=False)