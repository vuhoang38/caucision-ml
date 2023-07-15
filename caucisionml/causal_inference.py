import pandas as pd
import numpy as np
from dowhy import CausalModel
from dowhy.causal_identifier.identified_estimand import IdentifiedEstimand

from econml.metalearners import XLearner
from sklearn.preprocessing import OrdinalEncoder

from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, TweedieRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from sklearn import clone

models = {
    "Linear Regression": LinearRegression(),
    "XGBoost": XGBRegressor(),
    "Tweedie": MultiOutputRegressor(TweedieRegressor()),
    # "LightGBM": MultiOutputRegressor(LGBMRegressor()),
    "Random Forest": RandomForestRegressor(),
    "Support Vector Regression": MultiOutputRegressor(SVR()),
    "Decision Tree": DecisionTreeRegressor(),
    "Perceptron": MLPRegressor(),
    "K-nearest Neighbors": KNeighborsRegressor(),
}


def infer_from_project(
        df: pd.DataFrame,
        control_promotion: str,
        model_type: str,
        causal_graph: str,
) -> (pd.DataFrame, XLearner, OrdinalEncoder, IdentifiedEstimand, CausalModel):
    original_df = df

    categories = df["promotion"].unique()
    categories = np.delete(categories, np.argwhere(categories == control_promotion))
    categories = np.insert(categories, 0, control_promotion)

    df = df.dropna(subset=["outcome"])
    df = df.drop(columns=["user_id"])
    df["conversion"] = df["outcome"].apply(lambda x: 1 if x > 0 else 0)
    # TODO: Do we need to drop all NA columns?
    # TODO: Do we need to use OneHotEncoder, or is get_dummies enough?

    encoder = OrdinalEncoder(categories=[categories])
    encoder.fit(df[["promotion"]])
    df[["promotion"]] = encoder.transform(df[["promotion"]])

    model = CausalModel(
        data=df,
        treatment="promotion",
        outcome=["outcome", "conversion"],
        graph=causal_graph,
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    treatment = identified_estimand.treatment_variable
    outcome = identified_estimand.outcome_variable
    estimating_instrument_names = identified_estimand.instrumental_variables

    effect_modifier_names = model._graph.get_effect_modifiers(
        identified_estimand.treatment_variable, identified_estimand.outcome_variable
    )
    observed_common_causes_names = identified_estimand.get_backdoor_variables().copy()

    w_diff_x = [
        w for w in observed_common_causes_names if w not in effect_modifier_names
    ]

    if len(w_diff_x) > 0:
        effect_modifier_names.extend(w_diff_x)
    effect_modifiers = df[effect_modifier_names]
    # TODO: Think about whether we need to use drop_first?
    effect_modifiers = pd.get_dummies(effect_modifiers)

    if observed_common_causes_names:
        observed_common_causes = df[observed_common_causes_names]
        observed_common_causes = pd.get_dummies(observed_common_causes, drop_first=True)

    if estimating_instrument_names:
        estimating_instruments = df[estimating_instrument_names]
        estimating_instruments = pd.get_dummies(estimating_instruments, drop_first=True)

    X = effect_modifiers
    W = None  # common causes/ confounders
    Z = None  # Instruments
    Y = df[outcome]
    T = df[treatment]

    if observed_common_causes_names:
        W = observed_common_causes
    if estimating_instrument_names:
        Z = estimating_instruments

    # TODO: Implement cross validation to choose best estimators
    best_model_type, training_results = select_model(X, Y, T, model_type)
    est = XLearner(models=clone(models[best_model_type]))
    est.fit(Y, T, X=X)

    user_effects = original_df
    prepared_df = original_df.drop(columns=["user_id", "promotion", "outcome"])
    prepared_df = pd.get_dummies(prepared_df[effect_modifier_names])

    for index, category in enumerate(categories[1:]):
        effect = est.effect(prepared_df, T1=index + 1)
        framed_effect = pd.DataFrame(
            effect, columns=[f"{category} outcome", f"{category} conversion"]
        )
        user_effects = pd.concat([user_effects, framed_effect], axis=1)

    return user_effects, est, encoder, identified_estimand, model, categories, training_results


def infer_from_campaign_data(df, identified_estimand, categories, causal_model, est):
    original_df = df

    effect_modifier_names = causal_model._graph.get_effect_modifiers(
        identified_estimand.treatment_variable, identified_estimand.outcome_variable
    )
    observed_common_causes_names = identified_estimand.get_backdoor_variables().copy()

    w_diff_x = [
        w for w in observed_common_causes_names if w not in effect_modifier_names
    ]

    if len(w_diff_x) > 0:
        effect_modifier_names.extend(w_diff_x)

    user_effects = original_df
    prepared_df = original_df.drop(columns=["user_id"])
    prepared_df = pd.get_dummies(prepared_df[effect_modifier_names])

    for index, category in enumerate(categories[1:]):
        effect = est.effect(prepared_df, T1=index + 1)
        framed_effect = pd.DataFrame(
            effect, columns=[f"{category} outcome", f"{category} conversion"]
        )
        user_effects = pd.concat([user_effects, framed_effect], axis=1)

    return user_effects


def select_model(X, Y, T, model_type):
    dropped_Y = Y[['conversion']]
    concat_X = pd.concat([X, T], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(concat_X, dropped_Y, test_size=0.2, random_state=42)

    results = {}
    if model_type == 'Auto':
        for model_name, model in models.items():
            training_model = clone(model)

            start_time = time.time()
            training_model.fit(X_train, y_train)
            end_time = time.time()

            training_time = end_time - start_time

            y_pred = training_model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)

            results[model_name] = {
                'rmse': rmse, 'training_time': training_time,
            }
    else:
        model = clone(models[model_type])
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()

        training_time = end_time - start_time

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        results[model_type] = {
            'rmse': rmse, 'training_time': training_time,
        }

    best_model_type = min(results, key=lambda x: results[x]['rmse'])
    return best_model_type, results
