"""
Data loader that reproduces EXACTLY the dataset used
for supervised modeling in the notebook.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_and_preprocess(
    data_path="data/raw",
    test_size=0.2,
    random_state=42
):
   
    df_comp = pd.read_csv(f"{data_path}/crunchbase-companies.csv", encoding="latin1")
    df_rnd  = pd.read_csv(f"{data_path}/crunchbase-rounds.csv", encoding="latin1")
    df_acq  = pd.read_csv(f"{data_path}/crunchbase-acquisitions.csv", encoding="latin1")

    # Normalize permalinks
    for df in [df_comp, df_rnd, df_acq]:
        col = "permalink" if "permalink" in df.columns else "company_permalink"
        df[col] = df[col].astype(str).str.lower().str.strip()
        if col != "company_permalink":
            df.rename(columns={col: "company_permalink"}, inplace=True)

    df_comp = df_comp[df_comp["company_permalink"].str.contains("/company/")]

    
    df_rnd["funded_at"] = pd.to_datetime(df_rnd["funded_at"], errors="coerce")

    funding = df_rnd.groupby("company_permalink").agg(
        total_funding_usd=("raised_amount_usd", "sum"),
        avg_round_amount=("raised_amount_usd", "mean"),
        max_round_amount=("raised_amount_usd", "max"),
        num_funding_rounds=("company_permalink", "count"),
        first_funding_date=("funded_at", "min"),
        last_funding_date=("funded_at", "max"),
    ).reset_index()

    round_flags = (
        pd.get_dummies(
            df_rnd.set_index("company_permalink")["funding_round_type"],
            prefix="has"
        )
        .groupby(level=0)
        .max()
        .reset_index()
    )

    acquired_set = set(df_acq["company_permalink"].unique())

    df = (
        df_comp
        .merge(funding, on="company_permalink", how="left")
        .merge(round_flags, on="company_permalink", how="left")
    )

    flag_cols = [c for c in df.columns if c.startswith("has_")]
    df[flag_cols] = df[flag_cols].fillna(0).astype(int)

    df["total_funding_usd"] = df["total_funding_usd"].fillna(0)
    df["num_funding_rounds"] = df["num_funding_rounds"].fillna(0)

    df["is_acquired"]  = df["company_permalink"].isin(acquired_set).astype(int)
    df["is_ipo"]       = df["status"].eq("ipo").astype(int)
    df["is_closed"]    = df["status"].eq("closed").astype(int)
    df["is_operating"] = df["status"].eq("operating").astype(int)

    df["founded_at"] = pd.to_datetime(df["founded_at"], errors="coerce")

    SNAPSHOT_DATE = pd.to_datetime("2013-12-31")

    df["company_age_years"] = (
        (SNAPSHOT_DATE - df["founded_at"]).dt.days / 365.25
    )

    df["years_to_first_funding"] = (
        (df["first_funding_date"] - df["founded_at"]).dt.days / 365.25
    )

    df.loc[df["company_age_years"] < 0, "company_age_years"] = np.nan

    df = df.drop(
        columns=[
            "last_milestone_at",
            "founded_at",
            "founded_month",
            "founded_quarter",
        ],
        errors="ignore"
    )


    df["category_code"] = df["category_code"].fillna("not_defined")
    df["state_code"]    = df["state_code"].fillna("not_defined")
    df["city"]          = df["city"].fillna("other")
    df["region"]        = df["region"].fillna("other")


    df = df[df["funding_total_usd"].notna()]
    df = df[df["founded_year"].notna()]

    df["avg_round_amount"] = df["avg_round_amount"].fillna(0)
    df["max_round_amount"] = df["max_round_amount"].fillna(0)
    df["years_to_first_funding"] = df["years_to_first_funding"].fillna(-1)

   
    TOP_K = 20

    top_regions = df["region"].value_counts().nlargest(TOP_K).index
    top_cities  = df["city"].value_counts().nlargest(TOP_K).index

    df["region_grouped"] = df["region"].apply(
        lambda x: x if x in top_regions else "other"
    )

    df["city_grouped"] = df["city"].apply(
        lambda x: x if x in top_cities else "other"
    )

    df = df.drop(columns=["region", "city", "country_code"], errors="ignore")


    def make_binary_flags(df, column, top_k=20):
        top_values = df[column].value_counts().nlargest(top_k).index.tolist()
        df[column + "_bucket"] = df[column].apply(
            lambda x: x if x in top_values else "other"
        )
        dummies = pd.get_dummies(df[column + "_bucket"], prefix=f"has_{column}")
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(columns=[column, column + "_bucket"], errors="ignore")
        return df

    for col in ["category_code", "state_code", "region_grouped", "city_grouped"]:
        df = make_binary_flags(df, col, top_k=20)


    for col in ["first_funding_at", "last_funding_at"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.year

    df = df.drop(columns=["first_funding_date"], errors="ignore")

    numeric_cols = [
        "funding_total_usd",
        "avg_round_amount",
        "max_round_amount",
        "num_funding_rounds",
        "company_age_years",
        "years_to_first_funding",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].fillna(0)

    for col in df.columns:
        if col.startswith("has_"):
            df[col] = df[col].astype(int)

    # =========================================================
    # CLUSTERING + EXCLUDE CLUSTER 1
    # =========================================================
    X_cluster = df[numeric_cols].copy()
    scaler_cluster = StandardScaler()
    X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=4, random_state=random_state, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_cluster_scaled)

    df = df[df["cluster"] != 1].copy()
    df = df.drop(columns=["cluster"])

    # =========================================================
    # TARGET
    # =========================================================
    df["success"] = (
        (df["is_acquired"] == 1) |
        (df["is_ipo"] == 1)
    ).astype(int)

    # =========================================================
    # FEATURE SELECTION
    # =========================================================
    drop_cols = [
        "success",
        "is_acquired",
        "is_ipo",
        "is_closed",
        "is_operating",
        "last_funding_date",
        "company_permalink",
        "name",
        "status",
    ]

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["success"]

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test