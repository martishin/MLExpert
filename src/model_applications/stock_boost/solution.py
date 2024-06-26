import xgboost as xgb


def stock_boost(X_train, y_train, X_test):
    d_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    d_test = xgb.DMatrix(X_test, enable_categorical=True)
    stock_boost_model = xgb.train(
        {
            "objective": "binary:logistic",
            "tree_method": "exact",
            "max_cat_to_onehot": 11,
            "eta": 0.32,
            "max_depth": 7,
        },
        d_train,
    )
    raw_predictions = stock_boost_model.predict(d_test)
    threshold_predictions = [1 if value > 0.44 else 0 for value in raw_predictions]
    X_test["buy_signal"] = threshold_predictions
    y_test_predictions = X_test[["buy_signal"]].copy()
    X_test.drop("buy_signal", axis=1, inplace=True)
    return y_test_predictions
