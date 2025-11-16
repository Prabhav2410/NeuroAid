    # symptoms.py
    import os
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    import joblib

    # ---------------- Config ----------------
    CSV_PATH = "dataset_sorted.csv"
    CHUNK_SIZE = 20000
    MIN_SAMPLES_PER_DISEASE = 5
    RANDOM_STATE = 42

    # ---------------- Load dataset safely in chunks ----------------
    print("=" * 60)
    print("SAFE DATA LOADER")
    print("=" * 60)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ File not found: {CSV_PATH}")

    chunks = []
    used_encoding = None
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            for i, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False, encoding=enc, iterator=True)):
                print(f"Reading chunk {i+1} with encoding {enc} ... {len(chunk)} rows")
                chunks.append(chunk)
            used_encoding = enc
            break
        except Exception as e:
            chunks = []
            print(f"Failed reading with encoding {enc}: {e}")
    if not chunks:
        raise RuntimeError("Failed to read CSV with tested encodings.")

    df = pd.concat(chunks, ignore_index=True)
    print(f"✅ Loaded dataset successfully: {df.shape} (used encoding: {used_encoding})")

    # ---------------- Data cleaning ----------------
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    df = df.fillna(0)

    if 'diseases' not in df.columns:
        raise KeyError("Expected column 'diseases' in dataset_sorted.csv")

    symptom_cols = [col for col in df.columns if col != 'diseases']
    for col in symptom_cols:
        if df[col].dtype == 'object':
            print(f"⚠️ Converting non-numeric column to numeric: {col}")
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    original_size = len(df)
    df = df.drop_duplicates()
    print(f"Removed {original_size - len(df)} exact duplicates")

    disease_counts = df['diseases'].value_counts()
    valid_diseases = disease_counts[disease_counts >= MIN_SAMPLES_PER_DISEASE].index
    df = df[df['diseases'].isin(valid_diseases)].reset_index(drop=True)
    print(f"Kept {len(valid_diseases)} diseases with ≥{MIN_SAMPLES_PER_DISEASE} samples")

    # ---------------- Feature engineering ----------------
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    X = df.drop(columns=['diseases'])
    y = df['diseases']

    constant_features = [col for col in X.columns if X[col].nunique() == 1]
    if constant_features:
        print(f"⚠️ Removing {len(constant_features)} constant features")
        X = X.drop(columns=constant_features)

    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
    if to_drop:
        print(f"⚠️ Removing {len(to_drop)} highly correlated features")
        X = X.drop(columns=to_drop)

    print(f"Final feature count: {X.shape[1]}")

    # ---------------- Train-test split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # ---------------- Model training & SAFE CV ----------------
    print("\n" + "=" * 60)
    print("MODEL TRAINING (SAFE CV)")
    print("=" * 60)

    # If memory is tight, reduce n_estimators to 100 (increase later for final fit)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,            # allow parallelism for final fit
        random_state=RANDOM_STATE,
        oob_score=True
    )

    # determine safe CV folds based on smallest class count
    min_class_count = y_train.value_counts().min()
    safe_cv = min(5, int(min_class_count))
    if safe_cv < 2:
        safe_cv = 2
    if safe_cv != 5:
        print(f"⚠️ Adjusting CV folds from 5 -> {safe_cv} (smallest class has {min_class_count} samples)")

    # Run CV single-threaded to avoid forking many heavy workers (prevents OOM / SIGKILL)
    cv_scores = None
    try:
        print(f"Performing {safe_cv}-fold stratified cross-validation (single process)...")
        cv = StratifiedKFold(n_splits=safe_cv, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=1)
        print(f"CV F1 Scores: {cv_scores}")
        print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    except Exception as e:
        print("⚠️ Cross-validation failed or was aborted:", e)
        print("Proceeding to fit final model without CV.")
        cv_scores = [0.0]

    print("Fitting final model on full training set...")
    # if memory still an issue, temporarily set n_estimators=100 before .fit()
    # model.set_params(n_estimators=100)
    model.fit(X_train, y_train)
    if hasattr(model, "oob_score_"):
        try:
            print(f"OOB Score: {model.oob_score_:.4f}")
        except Exception:
            pass

    # ---------------- Evaluation ----------------
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    if train_accuracy - test_accuracy > 0.1:
        print("⚠️ WARNING: Possible overfitting detected! Consider changing hyperparameters.")

    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print(f"\nTraining F1: {train_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, y_test_pred, zero_division=0))

    # ---------------- Feature importance ----------------
    print("\n" + "=" * 60)
    print("TOP 20 IMPORTANT SYMPTOMS")
    print("=" * 60)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(feature_importance.head(20).to_string(index=False))

    # ---------------- Error analysis ----------------
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    conf_matrix = confusion_matrix(y_test, y_test_pred)
    classes = model.classes_

    errors = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and conf_matrix[i][j] > 0:
                errors.append({'true': classes[i], 'predicted': classes[j], 'count': conf_matrix[i][j]})

    errors_df = pd.DataFrame(errors).sort_values('count', ascending=False)
    print("\nTop 10 Misclassifications:")
    print(errors_df.head(10).to_string(index=False))

    # ---------------- Save artifacts ----------------
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    joblib.dump(model, 'rf_disease_model.pkl')
    joblib.dump(list(X.columns), 'symptom_list.pkl')
    joblib.dump(list(model.classes_), 'class_names.pkl')
    joblib.dump(feature_importance, 'feature_importance.pkl')

    metadata = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_f1_mean': float(np.mean(cv_scores)) if cv_scores is not None else 0.0,
        'cv_f1_std': float(np.std(cv_scores)) if cv_scores is not None else 0.0,
        'n_diseases': len(model.classes_),
        'n_features': len(X.columns),
        'n_samples': len(df)
    }
    joblib.dump(metadata, 'model_metadata.pkl')

    print("✅ Model and artifacts saved:")
    print("   - rf_disease_model.pkl")
    print(f"   - symptom_list.pkl ({len(X.columns)} symptoms)")
    print(f"   - class_names.pkl ({len(model.classes_)} diseases)")
    print("   - feature_importance.pkl")
    print("   - model_metadata.pkl")
    print("\nTRAINING COMPLETE")
