import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

model: MLPRegressor = None
scaler: StandardScaler = None
feature_columns: list = []

MODEL_PATH  = "ann_model.pkl"
SCALER_PATH = "scaler.pkl"
COLS_PATH   = "feature_cols.pkl"

# 2. DATA LOADING & PREPROCESSING

def load_and_preprocess(csv_path: str = "loan_data.csv"):
   
    df = pd.read_csv(csv_path)

    # Rename columns for clarity
    df.rename(columns={
        "loan_amnt":      "Principal",
        "person_income":  "Income",
        "loan_int_rate":  "InterestRate",
    }, inplace=True)

    # Drop rows where target is missing
    df.dropna(subset=["InterestRate"], inplace=True)

    # Remove obvious outliers (age > 100 or emp_exp > 60)
    df = df[df["person_age"] <= 100]
    df = df[df["person_emp_exp"] <= 60]

    # Fill remaining missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical columns
    cat_cols = ["person_gender", "person_education",
                "person_home_ownership", "loan_intent",
                "previous_loan_defaults_on_file"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Select features
    feature_cols = [
        "Principal",            # loan amount
        "Income",               # monthly-equivalent income
        "credit_score",         # credit score
        "person_age",           # age
        "person_emp_exp",       # employment experience (years)
        "loan_percent_income",  # loan-to-income ratio
        "person_home_ownership",# encoded home ownership
        "loan_intent",          # encoded loan purpose
        "previous_loan_defaults_on_file",  # encoded default history
        "cb_person_cred_hist_length",       # credit history length
    ]

    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df["InterestRate"].values

    return X, y, feature_cols


# 3. MODEL TRAINING

def train_model(csv_path: str = "loan_data.csv"):
    print("Loading & preprocessing data …")
    X, y, feat_cols = load_and_preprocess(csv_path)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc  = sc.transform(X_test)

    print(f"Training on {X_train_sc.shape[0]} samples, "
          f"testing on {X_test_sc.shape[0]} samples …")

    # ── ANN: Feed-Forward Neural Network ──────────────────────────────────
    ann = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),   # 3 hidden layers
        activation="relu",                  # ReLU activation
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )

    ann.fit(X_train_sc, y_train)

    # Evaluate
    y_pred = ann.predict(X_test_sc)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"Test MAE : {mae:.4f}%")
    print(f"Test R²  : {r2:.4f}")
    print(f"Training loss (final): {ann.loss_:.6f}")

    # Save artifacts
    joblib.dump(ann,       MODEL_PATH)
    joblib.dump(sc,        SCALER_PATH)
    joblib.dump(feat_cols, COLS_PATH)
    print("Model artifacts saved.")

    return ann, sc, feat_cols


# ─────────────────────────────────────────────
# 4. LOAD OR TRAIN
# ─────────────────────────────────────────────
def _initialise(csv_path: str = "loan_data.csv"):
    global model, scaler, feature_columns

    if (os.path.exists(MODEL_PATH) and
            os.path.exists(SCALER_PATH) and
            os.path.exists(COLS_PATH)):
        model          = joblib.load(MODEL_PATH)
        scaler         = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(COLS_PATH)
        print("Loaded pre-trained model from disk.")
    else:
        model, scaler, feature_columns = train_model(csv_path)


# ─────────────────────────────────────────────
# 5. PREDICTION FUNCTION  (called by app.py)
# ─────────────────────────────────────────────
def predict_interest_rate(input_data: dict, csv_path: str = "loan_data.csv") -> dict:
   
    global model, scaler, feature_columns

    # Initialise on first call
    if model is None:
        _initialise(csv_path)

    # Auto-compute loan_percent_income if not provided
    if input_data.get("loan_percent_income", 0) == 0 and input_data["Income"] > 0:
        input_data["loan_percent_income"] = (
            input_data["Principal"] / input_data["Income"]
        )

    # Build feature vector in correct order
    row = [input_data.get(col, 0) for col in feature_columns]
    X   = np.array(row).reshape(1, -1)
    X_sc = scaler.transform(X)

    # Predict
    rate = float(ann_predict(X_sc))
    rate = max(3.0, min(25.0, rate))   # clamp to realistic range

    # ── Confidence heuristic based on credit score ──────────────────────
    cs = input_data.get("credit_score", 600)
    confidence = round(min(98, max(60, 60 + (cs - 300) / 550 * 38)), 1)

    # ── EMI calculation (standard amortisation) ──────────────────────────
    principal    = input_data["Principal"]
    term_months  = input_data.get("term_months", 36)
    monthly_rate = rate / 12 / 100

    if monthly_rate > 0:
        emi = (principal * monthly_rate * (1 + monthly_rate) ** term_months) / \
              ((1 + monthly_rate) ** term_months - 1)
    else:
        emi = principal / term_months

    total_repayment = emi * term_months
    total_interest  = total_repayment - principal

    # ── Key driver ───────────────────────────────────────────────────────
    if cs >= 700:
        driver = f"High credit score ({cs}) reduced your predicted rate."
    elif cs < 550:
        driver = f"Low credit score ({cs}) increased your predicted rate."
    elif input_data.get("previous_loan_defaults_on_file", 0) == 1:
        driver = "Previous loan default raised your predicted rate."
    else:
        driver = "Loan-to-income ratio is the primary rate driver."

    return {
        "predicted_rate":  round(rate, 2),
        "confidence":      confidence,
        "monthly_emi":     round(emi, 0),
        "total_interest":  round(total_interest, 0),
        "total_repayment": round(total_repayment, 0),
        "key_driver":      driver,
    }


def ann_predict(X_scaled):
    return model.predict(X_scaled)[0]

# 6. STANDALONE TRAINING ENTRY-POINT
if __name__ == "__main__":
    train_model("loan_data.csv")
