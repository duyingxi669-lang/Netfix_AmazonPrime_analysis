from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator, ClusteringEvaluator
import pandas as pd

# 启动 Spark
spark = SparkSession.builder.appName("AmazonML").getOrCreate()

# 读取已清洗的数据
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("quote", '"') \
    .option("escape", '"') \
    .option("multiLine", "true") \
    .csv('D:/DPWII project/Amazon Prime dataset/amazon_cleaned.csv')

# 数值类型转换
df = df.withColumn("release_year", F.col("release_year").cast(IntegerType()))
df = df.withColumn("runtime", F.col("runtime").cast(IntegerType()))
df = df.withColumn("actor_count", F.col("actor_count").cast(IntegerType()))
df = df.withColumn("imdb_score", F.col("imdb_score").cast(FloatType()))
df = df.withColumn("tmdb_score", F.col("tmdb_score").cast(FloatType()))
df = df.withColumn("tmdb_popularity", F.col("tmdb_popularity").cast(FloatType()))

# ============================================================
# 聚类分析（K-Means）
# ============================================================
print("\n========== Cluster analysis ==========")

# 1. 选取数值特征
cluster_cols = ["imdb_score", "tmdb_score", "tmdb_popularity", "runtime", "actor_count", "release_year"]

# 2. 向量化
assembler = VectorAssembler(inputCols=cluster_cols, outputCol="features_raw")
feature_vector = assembler.transform(df).select("id", "title", "type", "main_genre", *cluster_cols, "features_raw")

# 3. 标准化
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
scaler_model = scaler.fit(feature_vector)
feature_vector = scaler_model.transform(feature_vector)

# 4. 用轮廓系数确定最佳 K 值
silhouette = []
for k in range(2, 20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(feature_vector)
    predictions = model.transform(feature_vector)
    evaluator = ClusteringEvaluator(featuresCol="features")
    silhouette.append(evaluator.evaluate(predictions))
    print(f"k = {k}, Silhouette = {silhouette[k-2]:.4f}")

# 5. 选定最佳 K，运行最终聚类
best_k = 6
kmeans = KMeans().setK(best_k).setSeed(1).setFeaturesCol("features")
model = kmeans.fit(feature_vector)
df_clustered = model.transform(feature_vector)

# 6. 查看每个簇的样本数量和平均特征
print("Number of samples in each cluster: ")
df_clustered.groupBy("prediction").count().orderBy("prediction").show()

print("Average characteristics of each cluster: ")
df_clustered.groupBy("prediction") \
    .avg(*cluster_cols) \
    .orderBy("prediction") \
    .show()

# 7. 保存聚类结果
df_clustered.select("id", "title", "type", "main_genre", "prediction", *cluster_cols) \
    .toPandas() \
    .to_csv("output/amazon_clustered.csv", index=False)

print("The clustering results have been saved.")

# ============================================================
# 分类分析（多特征）：预测影视剧类型（MOVIE/TV SHOW）
# ============================================================
print("\n========== Classification Analysis ==========")

# 1. 选取特征列
categorical_cols = ["main_genre", "age_certification"]
numeric_cols = ["imdb_score", "tmdb_score", "tmdb_popularity", "runtime", "actor_count", "release_year"]

# 检查缺失值
df = df.na.fill({"main_genre": "No Data", "age_certification": "No Data"})
df = df.na.fill(0, subset=numeric_cols)  # 数值列缺失填0

# 2. 编码 label
label_indexer = StringIndexer(inputCol="type", outputCol="label")

# 3. 对分类特征做索引 + 独热编码
indexers = [StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c + "_index", outputCol=c + "_vec") for c in categorical_cols]

# 4. 向量化所有特征
all_feature_cols = [c + "_vec" for c in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols=all_feature_cols, outputCol="features")

# 5. 拆分训练集和测试集
train_data, test_data = df.randomSplit([0.7, 0.3], seed=42)

# 6. 定义三个分类器
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20, seed=42)
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
gbt = GBTClassifier(featuresCol="features", labelCol="label", maxIter=20, seed=42)

# 7. 构建 Pipeline（编码 + 分类器）
def build_pipeline(classifier):
    stages = [label_indexer] + indexers + encoders + [assembler, classifier]
    return Pipeline(stages=stages)

# 8. 训练并评估
models = {"Random Forest": rf, "Logistic Regression": lr, "GBT": gbt}
results = []

for name, classifier in models.items():
    pipeline = build_pipeline(classifier)
    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)
    
    # 评估准确率
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = acc_evaluator.evaluate(predictions)
    
    # 评估 F1 分数
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )
    f1 = f1_evaluator.evaluate(predictions)
    
    results.append((name, accuracy, f1))
    print(f"{name} -> Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# === 参数调优、交叉验证 ===
print("\n--- GBT parameter tuning ---")

gbt_tuned = GBTClassifier(featuresCol="features", labelCol="label", seed=42)
pipeline_gbt = build_pipeline(gbt_tuned)

paramGrid = ParamGridBuilder() \
    .addGrid(gbt_tuned.maxDepth, [5, 10, 15]) \
    .addGrid(gbt_tuned.maxIter, [10, 20]) \
    .build()

crossval = CrossValidator(
    estimator=pipeline_gbt,
    estimatorParamMaps=paramGrid,
    evaluator=MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1"),
    numFolds=3,
    seed=42
)

# 调参
cvModel = crossval.fit(train_data)

# 查看所有参数组合的交叉验证平均F1得分
print("\nAverage F1 score for all parameter combinations: ")
for i, params in enumerate(paramGrid):
    param_str = ", ".join([f"{k.name}={v}" for k, v in params.items()])
    print(f"Combination{i+1}: {param_str} -> Average F1 score = {cvModel.avgMetrics[i]:.4f}")

# 打印最优参数
best_maxDepth = cvModel.bestModel.stages[-1]._java_obj.getMaxDepth()
best_maxIter = cvModel.bestModel.stages[-1]._java_obj.getMaxIter()
print(f"\nThe best parameter: maxDepth = {best_maxDepth}, maxIter = {best_maxIter}")

# 在测试集上评估
cv_predictions = cvModel.transform(test_data)

cv_acc = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
).evaluate(cv_predictions)
cv_f1 = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
).evaluate(cv_predictions)

results.append(("GBT (tuned)", cv_acc, cv_f1))
print(f"GBT (tuned) -> Accuracy: {cv_acc:.4f}, F1: {cv_f1:.4f}")

# 9. 保存对比结果
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1"])
results_df.to_csv("output/amazon_classification_compare.csv", index=False)
print("The results of the classification model comparison have been saved.")

# 10. 用最佳模型预测
best_model = models["GBT"]
pipeline = build_pipeline(best_model)
final_model = pipeline.fit(train_data)
sample_predictions = final_model.transform(test_data).select("id", "title", "type", "label", "prediction")
sample_predictions.show(10)
sample_predictions.toPandas().to_csv("output/amazon_classification_sample.csv", index=False)

# ============================================================
# 回归分析：预测imdb评分
# ============================================================
print("\n========== Regression analysis ==========")

# 1. 选取特征列
categorical_cols = ["main_genre", "age_certification"]
numeric_cols = ["runtime", "release_year", "tmdb_popularity", "actor_count", "tmdb_score"]

# 2. 缺失值处理
df = df.na.fill({"main_genre": "No Data", "age_certification": "No Data"})
df = df.na.fill(0, subset=numeric_cols)
df = df.filter(F.col("imdb_score").isNotNull())

# 3. 编码分类特征
indexers = [StringIndexer(inputCol=c, outputCol=c + "_index", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c + "_index", outputCol=c + "_vec") for c in categorical_cols]

# 4. 向量化所有特征
all_feature_cols = [c + "_vec" for c in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols=all_feature_cols, outputCol="features")

# 5. 构建 Pipeline
def build_feature_pipeline():
    return Pipeline(stages=indexers + encoders + [assembler])

feature_pipeline = build_feature_pipeline()
df_transformed = feature_pipeline.fit(df).transform(df).select("features", "imdb_score", "title", "id")

# 重命名目标列为 "label"
df_final = df_transformed.withColumnRenamed("imdb_score", "label").select("features", "label")

# 6. 拆分训练集和测试集
train_data, test_data = df_final.randomSplit([0.7, 0.3], seed=42)

# 7. 模型构建
#=== 模型 1：LinearRegression ===
print("\n--- Linear Regression ---")

lr = LinearRegression(featuresCol="features", labelCol="label")
lrModel = lr.fit(train_data)

# 打印系数和截距
print("Coefficients: {}".format(lrModel.coefficients[:5]))  # 只显示前 5 个
print("Intercept: {}".format(lrModel.intercept))

# 在测试集上评估
lr_results = lrModel.evaluate(test_data)
print("RMSE: {}".format(lr_results.rootMeanSquaredError))
print("MSE: {}".format(lr_results.meanSquaredError))
print("R2: {}".format(lr_results.r2))

lr_r2 = lr_results.r2
lr_rmse = lr_results.rootMeanSquaredError

#=== 模型 2：RandomForestRegressor ===
print("\n--- Random Forest Regressor ---")

rf = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=20, seed=42)
rfModel = rf.fit(train_data)

# 预测
rf_predictions = rfModel.transform(test_data)

# 评估
rf_evaluator_rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rf_evaluator_r2 = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

rf_rmse = rf_evaluator_rmse.evaluate(rf_predictions)
rf_r2 = rf_evaluator_r2.evaluate(rf_predictions)

print("RMSE: {}".format(rf_rmse))
print("R2: {}".format(rf_r2))

# 查看 Random Forest 特征重要性
print("\nRandom Forest feature importance (top 10): ")
feature_names = all_feature_cols
importance = rfModel.featureImportances.toArray()
for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {name}: {imp:.4f}")

# 保存 Random Forest 特征重要性
categorical_cols = ["main_genre", "age_certification"]
numeric_cols = ["runtime", "release_year", "tmdb_popularity", "actor_count", "tmdb_score"]

expanded_names = []     #构建特征名
for c in categorical_cols:
    categories = [row[0] for row in df.select(c).distinct().collect()]
    for cat in sorted(categories, key=str):
        expanded_names.append(f"{c}={cat}")
expanded_names += numeric_cols

importance = rfModel.featureImportances.toArray()
rf_importance = pd.DataFrame({
    "Feature": expanded_names,
    "Importance": importance
}).sort_values("Importance", ascending=False)
rf_importance.to_csv("output/amazon_rf_importance.csv", index=False)

#=== 模型 3：GBTRegressor ===
print("\n--- GBT Regressor ---")

gbt = GBTRegressor(featuresCol="features", labelCol="label", maxIter=20, seed=42)
gbtModel = gbt.fit(train_data)

# 预测
gbt_predictions = gbtModel.transform(test_data)

# 评估
gbt_rmse = rf_evaluator_rmse.evaluate(gbt_predictions)
gbt_r2 = rf_evaluator_r2.evaluate(gbt_predictions)

print("RMSE: {}".format(gbt_rmse))
print("R2: {}".format(gbt_r2))

# 查看 GBT 特征重要性
print("\nGBT feature importance (top 10): ")
importance_gbt = gbtModel.featureImportances.toArray()
for name, imp in sorted(zip(feature_names, importance_gbt), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {name}: {imp:.4f}")

# 保存 GBT 特征重要性
importance_gbt = gbtModel.featureImportances.toArray()
gbt_importance = pd.DataFrame({
    "Feature": expanded_names,
    "Importance": importance_gbt
}).sort_values("Importance", ascending=False)
gbt_importance.to_csv("output/amazon_gbt_importance.csv", index=False)

# 8. 保存结果
results = [
    ("Linear Regression", lr_rmse, lr_r2),
    ("Random Forest", rf_rmse, rf_r2),
    ("GBT", gbt_rmse, gbt_r2)
]

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2"])
results_df.to_csv("output/amazon_regression_compare.csv", index=False)

print("\nComparison results of regression models: ")
for model, rmse, r2 in results:
    print(f"{model}: RMSE = {rmse:.4f}, R² = {r2:.4f}")

print("The results of the regression model comparison have been saved.")
spark.stop()