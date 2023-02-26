import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy.spatial import distance
from sklearn.decomposition import PCA

class ADAM:
    @classmethod
    def _run_extremes_imputation(cls, df_train: pd.DataFrame, continuous_features: list, side: str):
        df_std1 = df_train.copy(deep=True)
        df_std2 = df_train.copy(deep=True)
        df_std3 = df_train.copy(deep=True)

        for c in continuous_features:
            if side == "min":
                q = df_train.loc[:, c].dropna().quantile(q=0.25, interpolation='midpoint')
                mean_val = df_train[df_train[c] <= q].loc[:, c].mean(skipna=True)
                sign = -1.0
            elif side == "max":
                q = df_train.loc[:, c].dropna().quantile(q=0.75, interpolation='midpoint')
                mean_val = df_train[df_train[c] >= q].loc[:, c].mean(skipna=True)
                sign = 1.0
            else:
                raise ValueError("Invalid side parameter.")

            std_val = df_train.loc[:, c].std(skipna=True, ddof=1)
            md_rows = np.argwhere(np.isnan(df_train.loc[:, c].values)).flatten()
            dist_gen_1 = np.random.default_rng().normal(mean_val + (sign * std_val), 1, md_rows.shape[0])
            dist_gen_2 = np.random.default_rng().normal(mean_val + (sign * 2.0 * std_val), 1, md_rows.shape[0])
            dist_gen_3 = np.random.default_rng().normal(mean_val + (sign * 3.0 * std_val), 1, md_rows.shape[0])
            df_std1.loc[md_rows, c] = dist_gen_1
            df_std2.loc[md_rows, c] = dist_gen_2
            df_std3.loc[md_rows, c] = dist_gen_3

        return df_std1, df_std2, df_std3

    @classmethod
    def _calculate_factor(cls, dim_red_ds: list, i_col: int):
        col_1d = []
        for ds in dim_red_ds:
            col_1d.append(ds[i_col, 0])
        col_1d_dists = []
        for i in range(1, len(dim_red_ds)):
            col_1d_dists.append(distance.euclidean(col_1d[0], col_1d[i]))
        col_1d_dists = np.asarray(col_1d_dists)
        sum_dists = np.sum(col_1d_dists)
        col_1d_dists /= sum_dists
        factor = np.average(col_1d_dists, weights=np.arange(3, 0, -1))
        return factor

    @classmethod
    def adjust(cls, df_train: pd.DataFrame, df_test: pd.DataFrame, df_imputed: pd.DataFrame,
               categorical_features: list = None, missing_tail: str = "left"):
        if missing_tail not in ["left", "right"]:
            raise ValueError("Invalid missing tail.")
        if categorical_features is None:
            categorical_features = []

        all_columns = df_train.columns.values
        continuous_features = [f for f in all_columns if not f.startswith(tuple(categorical_features))]
        categorical_features = [f for f in all_columns if f.startswith(tuple(categorical_features))]
        is_asc = True if missing_tail == "left" else False

        df_std1_min, df_std2_min, df_std3_min, = \
            cls._run_extremes_imputation(df_train, continuous_features, side="min")
        df_std1_max, df_std2_max, df_std3_max, = \
            cls._run_extremes_imputation(df_train, continuous_features, side="max")

        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        base_df_imp = df_train.copy(deep=True)
        base_df_imp.loc[:, continuous_features] = \
            imp_mean.fit_transform(base_df_imp.loc[:, continuous_features])
        df_const_min = [df_std1_min, df_std2_min, df_std3_min]
        df_const_max = [df_std1_max, df_std2_max, df_std3_max]

        for ds in [base_df_imp] + df_const_min + df_const_max:
            imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            ds.loc[:, categorical_features] = imp_mode.fit_transform(ds.loc[:, categorical_features])

        feat_factor = {}
        for i_col, col in enumerate(all_columns):
            if col in continuous_features:
                dim_red_ds = []
                pca = PCA(n_components=1)
                df_const_arr = df_const_min if is_asc else df_const_max
                for ds in [base_df_imp] + df_const_arr:
                    dim_red_ds.append(pca.fit_transform(ds.T))
                factor = cls._calculate_factor(dim_red_ds, i_col)
                feat_factor[col] = -1.0 * factor if is_asc else factor

        df_imputed_adj = df_imputed.copy(deep=True)
        for col in continuous_features:
            md_rows = np.argwhere(np.isnan(df_test.loc[:, col].values)).flatten()
            df_imputed_adj.loc[md_rows, col] += feat_factor[col] * df_imputed.loc[md_rows, col]

        return df_imputed_adj
