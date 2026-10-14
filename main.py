import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

SEEDS = (42, 0, 1, 7, 123)
TERMS = ("very_low", "low", "medium", "high", "very_high")


def load_data():
    """Загрузить исходные признаки и медианную стоимость жилья по территориальным группам."""
    data = fetch_california_housing(as_frame=True)
    return data.data.copy(), data.target.copy()


def preprocess(X_train, X_test):
    """Заполнить пропуски и ограничить выбросы по медианам и 99-м процентилям обучающей выборки."""
    medians = X_train.median()
    if medians.isna().any():
        raise ValueError("Каждый обучающий признак должен содержать хотя бы одно наблюдаемое значение.")

    train = X_train.fillna(medians).copy()
    test = X_test.fillna(medians).copy()
    if not np.isfinite(train.to_numpy()).all() or not np.isfinite(test.to_numpy()).all():
        raise ValueError("После заполнения пропусков значения признаков должны быть конечными.")

    for column in ("AveRooms", "AveOccup", "AveBedrms", "Population"):
        if column in train.columns:
            cap = train[column].quantile(0.99)
            train[column] = train[column].clip(upper=cap)
            test[column] = test[column].clip(upper=cap)
    return train, test


class FuzzyFeaturizer:
    """Преобразовать каждый признак в пять треугольных термов с вершинами по процентилям обучающей выборки."""

    def __init__(self, feature_names):
        self.feature_names = tuple(feature_names)
        self.peaks_ = {}

    def fit(self, X: pd.DataFrame):
        values = X.loc[:, self.feature_names].to_numpy(dtype=float)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError("Для настройки функций принадлежности нужны непустые данные с конечными значениями.")

        percentiles = np.linspace(0, 100, len(TERMS))
        self.peaks_ = {
            feature: np.percentile(X[feature], percentiles)
            for feature in self.feature_names
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.peaks_:
            raise ValueError("Перед преобразованием данных настройте функции принадлежности методом fit.")
        if not np.isfinite(X.loc[:, self.feature_names].to_numpy(dtype=float)).all():
            raise ValueError("Входы функций принадлежности должны быть конечными.")

        columns = []
        for feature in self.feature_names:
            peaks = self.peaks_[feature]
            values = np.clip(X[feature].to_numpy(dtype=float), peaks[0], peaks[-1])
            for i in range(len(TERMS)):
                if peaks[0] == peaks[-1]:
                    degree = np.full(len(values), float(i == len(TERMS) // 2))
                else:
                    triangle = [
                        peaks[max(i - 1, 0)],
                        peaks[i],
                        peaks[min(i + 1, len(TERMS) - 1)],
                    ]
                    degree = fuzz.trimf(values, triangle)
                columns.append(degree)
        return np.column_stack(columns)


def build_model(seed):
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=seed,
    )


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def explain_sample(featurizer, model, X_test, y_test, idx=0):
    """Описать нечёткие входы объекта, не приписывая им вклад в прогноз стоимости."""
    row = X_test.iloc[[idx]]
    encoded = featurizer.transform(row)
    prediction = model.predict(encoded)[0]

    print("\nЛингвистическое описание признаков примера:")
    term_labels = ("очень низкий", "низкий", "средний", "высокий", "очень высокий")
    degrees = encoded[0].reshape(len(featurizer.feature_names), len(TERMS))
    for feature, memberships in zip(featurizer.feature_names, degrees):
        best = int(np.argmax(memberships))
        print(
            f"  {feature:12s}: {term_labels[best]:13s} "
            f"(принадлежность={memberships[best]:.2f}, значение={row[feature].iloc[0]:.2f})"
        )
    print(
        f"  Медианная стоимость жилья территориальной группы: прогноз=${prediction * 100_000:,.0f}, "
        f"факт=${y_test.iloc[idx] * 100_000:,.0f}"
    )


def multiseed_eval(X, y, seeds):
    """Оценить обе модели на одинаковых разбиениях 80/20 с предобработкой по обучающей части."""
    rows = []
    for index, seed in enumerate(seeds):
        print(f"\nРазбиение {index + 1}/{len(seeds)} (seed={seed})", flush=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        X_train, X_test = preprocess(X_train, X_test)

        featurizer = FuzzyFeaturizer(X.columns).fit(X_train)
        print("  Обучение MLP на нечётких признаках...", flush=True)
        fuzzy_model = build_model(seed).fit(featurizer.transform(X_train), y_train)
        fuzzy_metrics = evaluate(y_test, fuzzy_model.predict(featurizer.transform(X_test)))

        scaler = StandardScaler().fit(X_train)
        print("  Обучение MLP на стандартизированных признаках...", flush=True)
        raw_model = build_model(seed).fit(scaler.transform(X_train), y_train)
        raw_metrics = evaluate(y_test, raw_model.predict(scaler.transform(X_test)))

        for name, metrics in (("Fuzzy-Neural", fuzzy_metrics), ("Raw-Baseline", raw_metrics)):
            rows.append({"seed": seed, "model": name, **metrics})
        if index == 0:
            explain_sample(featurizer, fuzzy_model, X_test, y_test)

    return pd.DataFrame(rows)


def main():
    print("Загрузка California Housing...", flush=True)
    X, y = load_data()
    print(f"Объектов: {len(X)}, исходных признаков: {X.shape[1]}, нечётких признаков: {X.shape[1] * len(TERMS)}")
    print("Единицы MAE/RMSE: 100 000 долларов. Скрытые слои MLP: 64, 32.")
    print("Параметры предобработки определяются только по обучающей части.")

    results = multiseed_eval(X, y, SEEDS)
    model_labels = {
        "Fuzzy-Neural": "Нечёткие признаки + MLP",
        "Raw-Baseline": "Исходные признаки + MLP",
    }
    print("\nМетрики на тестовой части каждого разбиения:")
    display = results.assign(model=results["model"].map(model_labels)).rename(
        columns={"seed": "Seed", "model": "Модель"}
    )
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print("\nСреднее +/- стандартное отклонение по разбиениям (ddof=0):")
    for name, group in results.groupby("model", sort=False):
        print(model_labels[name])
        for metric in ("MAE", "RMSE", "R2"):
            values = group[metric].to_numpy()
            print(f"  {metric:4s}: {values.mean():.4f} +/- {values.std():.4f}")


if __name__ == "__main__":
    main()
