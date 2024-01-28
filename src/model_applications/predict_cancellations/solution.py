from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler


def predict_cancellations(user_interaction_df):
    assembler = VectorAssembler(
        inputCols=[
            "month_interaction_count",
            "week_interaction_count",
            "day_interaction_count",
        ],
        outputCol="features",
    )

    features_df = assembler.transform(user_interaction_df)
    features_df = features_df.withColumn("label", features_df["cancelled_within_week"])

    lr_model = LogisticRegression(
        maxIter=10, threshold=0.6, elasticNetParam=1, regParam=0.1
    )
    trained_lr_model = lr_model.fit(features_df)

    predictions_df = trained_lr_model.transform(features_df)
    predictions_df = predictions_df.select(
        ["user_id", "rawPrediction", "probability", "prediction"]
    )

    return predictions_df
