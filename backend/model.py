import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import jellyfish
import re
import pickle
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class AMLNameMatcher:
    def __init__(self, dataset_path: str = 'cleaned_aml_data.xlsx'):
        """
        Système AML avec matching de noms.
        Ajouts :
          - Translittération arabe -> latin
          - Logistic Regression (en plus de RF et XGB)
          - Matching plus robuste + affichage de la translittération utilisée
        """
        self.dataset_path = dataset_path
        self.df = None
        self.rf_model = None
        self.xgb_model = None
        self.lr_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.risk_weights = {
            'Terrorism': 1.0,
            'Money Laundering': 0.9,
            'PEP': 0.7,
            'Government': 0.6,
            'Tech Investment': 0.4,
            'Real Estate': 0.5,
            'Donations': 0.3,
            'Offshore': 0.8
        }
        # table translittération arabe -> latin
        self._arabic_map = {
            'ا':'a','أ':'a','إ':'i','آ':'a','ب':'b','ت':'t','ث':'th','ج':'j','ح':'h','خ':'kh',
            'د':'d','ذ':'dh','ر':'r','ز':'z','س':'s','ش':'sh','ص':'s','ض':'d','ط':'t','ظ':'z',
            'ع':'a','غ':'gh','ف':'f','ق':'q','ك':'k','ل':'l','م':'m','ن':'n','ه':'h','ة':'a',
            'و':'w','ؤ':'u','ي':'y','ئ':'i','ى':'a','‎ٰ':'','ﻻ':'la','لا':'la','ٔ':'','ٕ':'',
            '٠':'0','١':'1','٢':'2','٣':'3','٤':'4','٥':'5','٦':'6','٧':'7','٨':'8','٩':'9'
        }

    # =========================
    # Chargement & Préparation
    # =========================
    def load_and_prepare_data(self):
        try:
            if self.dataset_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(self.dataset_path)
            else:
                self.df = pd.read_csv(self.dataset_path)

            print(f"Dataset chargé: {len(self.df)} entrées")
            print(f"Colonnes disponibles: {list(self.df.columns)}")

            self.df = self.df.dropna(subset=['Full Name'])
            self.create_robust_features()
            self.encode_categorical_features()
            self.create_balanced_labels()
            print("Données préparées avec succès")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
            return False

    def create_robust_features(self):
        self.df['name_length'] = self.df['Full Name'].str.len()
        denom = (self.df['name_length'].max() - self.df['name_length'].min())
        self.df['name_length_normalized'] = (self.df['name_length'] - self.df['name_length'].min()) / denom if denom != 0 else 0.0
        self.df['has_arabic'] = self.df['Full Name'].str.contains('[^\x00-\x7F]', na=False).astype(int)
        self.df['word_count'] = self.df['Full Name'].str.split().str.len()
        self.df['avg_word_length'] = self.df['Full Name'].apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) and len(str(x).split())>0 else 0
        )
        self.df['has_common_prefixes'] = self.df['Full Name'].str.contains(
            r'\b(Mr|Mrs|Dr|Prof|Al|Abu|Ibn)\b', case=False, na=False
        ).astype(int)
        self.df['name_length_bin'] = pd.cut(
            self.df['name_length'], bins=[0, 10, 20, 30, float('inf')],
            labels=['Short', 'Medium', 'Long', 'Very_Long']
        ).astype(str)

    def encode_categorical_features(self):
        categorical_columns = ['Nationality', 'Risk Category', 'Source of Funds', 'Watchlist Database', 'name_length_bin']
        for col in categorical_columns:
            if col in self.df.columns:
                value_counts = self.df[col].value_counts()
                rare_threshold = len(self.df) * 0.03
                rare_values = value_counts[value_counts < rare_threshold].index
                self.df[col] = self.df[col].fillna('Unknown')
                self.df.loc[self.df[col].isin(rare_values), col] = 'Other'
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"Encodage de la colonne: {col}")
                print(f"  Valeurs uniques: {len(le.classes_)}")
                print(f"  Valeurs rares regroupées: {len(rare_values)}")

    def create_balanced_labels(self):
        def assign_risk_label_fuzzy(row):
            risk_category = str(row.get('Risk Category', '')).lower()
            nationality = str(row.get('Nationality', '')).lower()
            source_funds = str(row.get('Source of Funds', '')).lower()
            risk_score = 0
            if 'terrorism' in risk_category:
                risk_score += 0.9
            elif 'money laundering' in risk_category or 'money la' in risk_category:
                risk_score += 0.8
            elif 'pep' in risk_category:
                risk_score += 0.6
            elif 'offshore' in risk_category:
                risk_score += 0.5
            high_risk_countries = ['syria', 'afghanistan', 'iran', 'north korea']
            medium_risk_countries = ['russia', 'china', 'turkey']
            if any(c in nationality for c in high_risk_countries):
                risk_score += 0.2
            elif any(c in nationality for c in medium_risk_countries):
                risk_score += 0.1
            if 'government' in source_funds and ('yes' in str(row.get('Watchlist Database', '')).lower()):
                risk_score += 0.15
            name_hash = hash(str(row.get('Full Name', ''))) % 100
            if name_hash < 5:
                risk_score += 0.1
            elif name_hash > 95:
                risk_score -= 0.1
            base_threshold = 0.7
            row_index = row.name if hasattr(row, 'name') else 0
            threshold_variation = (row_index % 10) * 0.01
            adaptive_threshold = base_threshold + threshold_variation - 0.05
            return 1 if risk_score >= adaptive_threshold else 0

        self.df['decision'] = self.df.apply(assign_risk_label_fuzzy, axis=1)
        self.add_controlled_label_noise()
        class_distribution = self.df['decision'].value_counts()
        print(f"\nDistribution des classes:")
        print(f"  Classe 0 (Allowed): {class_distribution.get(0, 0)} ({class_distribution.get(0, 0)/len(self.df)*100:.1f}%)")
        print(f"  Classe 1 (Blocked): {class_distribution.get(1, 0)} ({class_distribution.get(1, 0)/len(self.df)*100:.1f}%)")
        ratio = min(class_distribution) / max(class_distribution)
        if ratio > 0.95:
            print("⚠️  Distribution trop parfaite - Données probablement synthétiques")
        elif ratio < 0.1:
            print("⚠️  ATTENTION: Déséquilibre extrême des classes détecté!")

    def add_controlled_label_noise(self, noise_rate=0.05):
        n_samples = len(self.df)
        n_flip = int(n_samples * noise_rate)
        np.random.seed(42)
        flip_indices = np.random.choice(n_samples, n_flip, replace=False)
        original_decisions = self.df['decision'].copy()
        for idx in flip_indices:
            self.df.loc[self.df.index[idx], 'decision'] = 1 - self.df.loc[self.df.index[idx], 'decision']
        flipped = (original_decisions != self.df['decision']).sum()
        print(f"Labels modifiés pour réalisme: {flipped} ({flipped/n_samples*100:.1f}%)")

    def prepare_features_for_training(self):
        numeric_features = ['name_length_normalized', 'word_count', 'avg_word_length', 'has_arabic', 'has_common_prefixes']
        encoded_features = [c for c in self.df.columns if c.endswith('_encoded')]
        all_features = numeric_features + encoded_features
        available_features = [c for c in all_features if c in self.df.columns]
        print(f"Features sélectionnées: {available_features}")
        self.feature_names = available_features
        return self.df[available_features].fillna(0)

    def diagnose_dataset(self):
        print("\n=== DIAGNOSTIC DU DATASET ===")
        print(f"Taille totale: {len(self.df)} échantillons")
        class_counts = self.df['decision'].value_counts()
        print(f"Distribution des classes: {dict(class_counts)}")
        duplicates = self.df.duplicated(subset=['Full Name']).sum()
        print(f"Noms en doublon: {duplicates}")
        categorical_cols = [c for c in self.df.columns if c.endswith('_encoded')]
        for col in categorical_cols:
            unique_vals = self.df[col].nunique()
            print(f"{col}: {unique_vals} valeurs uniques")
            if unique_vals == len(self.df):
                print(f"  ⚠️  {col} a autant de valeurs que d'échantillons - Feature ID potentielle!")
        X = self.prepare_features_for_training()
        y = self.df['decision']
        correlations = []
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                corr = X[col].corr(y)
                correlations.append((col, 0.0 if pd.isna(corr) else abs(corr)))
                if abs(corr) > 0.95:
                    print(f"  ⚠️  {col} corrélation très élevée avec target: {corr:.4f}")
        correlations.sort(key=lambda x: x[1], reverse=True)
        print(f"\nTop corrélations avec target:")
        for col, corr in correlations[:3]:
            print(f"  {col}: {corr:.4f}")
        return correlations

    # =========================
    # Entraînement des modèles
    # =========================
    def train_models(self, test_size=0.3, use_cross_validation=True):
        if self.df is None:
            print("Veuillez d'abord charger les données")
            return False

        correlations = self.diagnose_dataset()
        X = self.prepare_features_for_training()
        y = self.df['decision']

        high_corr_features = [col for col, corr in correlations if corr > 0.9 and col in X.columns]
        if high_corr_features:
            print(f"\n⚠️  Suppression des features trop corrélées: {high_corr_features}")
            X = X.drop(columns=high_corr_features)
            self.feature_names = [f for f in self.feature_names if f not in high_corr_features]

        if len(np.unique(y)) < 2:
            print("⚠️  Une seule classe présente dans les données!")
            return False
        if len(X.columns) < 2:
            print("⚠️  Pas assez de features après filtrage!")
            return False

        actual_test_size = min(0.4, max(0.2, test_size))
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=actual_test_size, random_state=42, stratify=y
        )

        # léger bruit sur TRAIN uniquement
        X_train_noisy = X_train_raw.copy()
        noise_scale = 0.01
        for col in X_train_noisy.select_dtypes(include=[np.number]).columns:
            std_col = X_train_noisy[col].std()
            if std_col > 0:
                X_train_noisy[col] = X_train_noisy[col] + np.random.normal(0, noise_scale * std_col, len(X_train_noisy))

        # scaling
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train_noisy), columns=X_train_noisy.columns, index=X_train_noisy.index)
        X_test = pd.DataFrame(self.scaler.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)

        # split train/val pour XGB
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        print(f"Taille d'entraînement: {len(X_tr)}")
        print(f"Taille de validation: {len(X_val)}")
        print(f"Taille de test: {len(X_test)}")

        # RF
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=42,
            class_weight='balanced',
            bootstrap=True,
            oob_score=True
        )

        # XGB
        n_pos = (y_tr == 1).sum()
        n_neg = (y_tr == 0).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False
        )

        # Logistic Regression
        self.lr_model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            solver='lbfgs'
        )

        # Entraînements
        try:
            self.rf_model.fit(X_tr, y_tr)
            print("Random Forest entraîné")
        except Exception as e:
            print(f"Erreur Random Forest: {e}")

        try:
            early_stop_cb = xgb.callback.EarlyStopping(rounds=50, save_best=True, maximize=False)
            self.xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[early_stop_cb], verbose=False)
            print("XGBoost entraîné (early stopping)")
        except TypeError:
            self.xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            print("XGBoost entraîné (fallback sans early stopping)")
        except Exception as e:
            print(f"Erreur XGBoost: {e}")
            self.xgb_model.fit(X_tr, y_tr)

        try:
            self.lr_model.fit(X_tr, y_tr)
            print("Logistic Regression entraînée")
        except Exception as e:
            print(f"Erreur Logistic Regression: {e}")

        # CV
        if use_cross_validation and len(X_train) >= 10:
            self.evaluate_with_cross_validation(X_train, y_train)

        # Test
        self.evaluate_on_test_set(X_test, y_test)
        self.show_feature_importance()

        # Overfitting check
        for name, model in [('RF', self.rf_model), ('XGB', self.xgb_model), ('LR', self.lr_model)]:
            tr_pred = model.predict(X_tr)
            te_pred = model.predict(X_test)
            tr_acc = accuracy_score(y_tr, tr_pred)
            te_acc = accuracy_score(y_test, te_pred)
            print(f"{name} - Train: {tr_acc:.4f}, Test: {te_acc:.4f}, Écart: {abs(tr_acc-te_acc):.4f}")

        self._last_test = (X_test, y_test)
        return True

    def evaluate_with_cross_validation(self, X, y, cv_folds=5):
        print(f"\n=== VALIDATION CROISÉE ({cv_folds} folds) ===")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        rf_scores = cross_val_score(self.rf_model, X, y, cv=skf, scoring='accuracy')
        print(f"Random Forest CV Accuracy: {rf_scores.mean():.4f} (+/- {rf_scores.std()*2:.4f})")
        xgb_scores = cross_val_score(self.xgb_model, X, y, cv=skf, scoring='accuracy')
        print(f"XGBoost CV Accuracy: {xgb_scores.mean():.4f} (+/- {xgb_scores.std()*2:.4f})")
        lr_scores = cross_val_score(self.lr_model, X, y, cv=skf, scoring='accuracy')
        print(f"LogReg  CV Accuracy: {lr_scores.mean():.4f} (+/- {lr_scores.std()*2:.4f})")

    def evaluate_on_test_set(self, X_test, y_test):
        print(f"\n=== ÉVALUATION TEST SET ===")
        preds = {
            'RF': self.rf_model.predict(X_test),
            'XGB': self.xgb_model.predict(X_test),
            'LR': self.lr_model.predict(X_test),
        }
        probas = {
            'RF': self.rf_model.predict_proba(X_test)[:,1],
            'XGB': self.xgb_model.predict_proba(X_test)[:,1],
            'LR': self.lr_model.predict_proba(X_test)[:,1],
        }

        for name in ['RF', 'XGB', 'LR']:
            print(f"{name} Test Accuracy: {accuracy_score(y_test, preds[name]):.4f}")

        if hasattr(self.rf_model, 'oob_score_'):
            print(f"Random Forest OOB Score: {self.rf_model.oob_score_:.4f}")

        for name in ['RF', 'XGB', 'LR']:
            print(f"\n=== RAPPORT {name} ===")
            print(classification_report(y_test, preds[name]))

        print(f"\n=== MATRICES DE CONFUSION ===")
        for name in ['RF', 'XGB', 'LR']:
            print(f"{name}:")
            print(confusion_matrix(y_test, preds[name]))

        try:
            print("\nScores additionnels :")
            print(f"RF  ROC-AUC: {roc_auc_score(y_test, probas['RF']):.4f}")
            print(f"XGB ROC-AUC: {roc_auc_score(y_test, probas['XGB']):.4f}")
            print(f"LR  ROC-AUC: {roc_auc_score(y_test, probas['LR']):.4f}")
            print(f"RF  PR-AUC:  {average_precision_score(y_test, probas['RF']):.4f}")
            print(f"XGB PR-AUC: {average_precision_score(y_test, probas['XGB']):.4f}")
            print(f"LR  PR-AUC:  {average_precision_score(y_test, probas['LR']):.4f}")
        except Exception:
            pass

    def show_feature_importance(self):
        print(f"\n=== IMPORTANCE DES FEATURES ===")
        if self.rf_model and len(self.feature_names) > 0:
            rf_imp = self.rf_model.feature_importances_
            rf_features = sorted(zip(self.feature_names, rf_imp), key=lambda x: x[1], reverse=True)[:5]
            print("Random Forest Top Features:")
            for f, v in rf_features:
                print(f"  {f}: {v:.4f}")
        if self.xgb_model and len(self.feature_names) > 0:
            xgb_imp = self.xgb_model.feature_importances_
            xgb_features = sorted(zip(self.feature_names, xgb_imp), key=lambda x: x[1], reverse=True)[:5]
            print("\nXGBoost Top Features:")
            for f, v in xgb_features:
                print(f"  {f}: {v:.4f}")
        if self.lr_model and len(self.feature_names) > 0:
            coefs = np.abs(self.lr_model.coef_[0])
            lr_features = sorted(zip(self.feature_names, coefs), key=lambda x: x[1], reverse=True)[:5]
            print("\nLogistic Regression Top Features (|coef|):")
            for f, v in lr_features:
                print(f"  {f}: {v:.4f}")

    # =========================
    # Normalisation & Matching
    # =========================
    def _strip_arabic_diacritics(self, s: str) -> str:
        return re.sub(r'[\u064B-\u0652]', '', s)

    def arabic_to_latin(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = self._strip_arabic_diacritics(text)
        text = text.replace('أ','ا').replace('إ','ا').replace('آ','ا')
        out = []
        for ch in text:
            out.append(self._arabic_map.get(ch, ch))
        return ''.join(out)

    def _strip_prefixes_in_token(self, token: str) -> str:
        # enlève al/el/ad/ed/ibn/bin/ben même s'ils sont collés au début
        return re.sub(r'^(al|el|ad|ed|ibn|bin|ben)[\-\_]*', '', token)

    def _canonicalize_latin(self, s: str) -> str:
        """Canonisation tolérante des translittérations arabes en latin."""
        s = s.lower()
        # normaliser diacritiques latins
        trans = str.maketrans({
            'à':'a','á':'a','â':'a','ä':'a','å':'a',
            'è':'e','é':'e','ê':'e','ë':'e',
            'î':'i','ï':'i','ì':'i','í':'i',
            'ô':'o','ö':'o','ò':'o','ó':'o',
            'û':'u','ü':'u','ù':'u','ú':'u',
            'ç':'s','ñ':'n','ý':'y','ÿ':'y'
        })
        s = s.translate(trans)
        # nettoyer ponctuation douce -> espace
        s = re.sub(r'[^\w\s]', ' ', s)
        # traiter token par token (enlever préfixes collés)
        tokens = [t for t in re.split(r'\s+', s) if t]
        tokens = [self._strip_prefixes_in_token(t) for t in tokens]
        s = ' '.join(tokens)
        # unifier digrammes arabes
        s = (s.replace('gh','g')
               .replace('kh','k')
               .replace('th','t')
               .replace('dh','d')
               .replace('sh','s')
               .replace('ch','s'))
        # compresser voyelles successives → 'a'
        s = re.sub(r'[aeiouy]+', 'a', s)
        # enlever doubles consonnes
        s = re.sub(r'(bb|cc|dd|ff|gg|hh|jj|kk|ll|mm|nn|pp|qq|rr|ss|tt|vv|ww|xx|zz)',
                   lambda m: m.group(0)[0], s)
        # retirer non-alphanumérique et espaces
        s = re.sub(r'[^a-z0-9 ]+', '', s)
        s = re.sub(r'\s+', '', s).strip()
        return s

    def _phonetic_code(self, s: str) -> str:
        try:
            return jellyfish.metaphone(s)
        except Exception:
            return s

    def normalize_name(self, name: str) -> str:
        if pd.isna(name):
            return ""
        s = str(name)
        # translittérer si contient arabe
        if re.search(r'[\u0600-\u06FF]', s):
            s = self.arabic_to_latin(s)
        s = s.lower()
        s = re.sub(r'[^\w\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def _token_set(self, s: str):
        return set([t for t in re.split(r'\s+', s) if t])

    def calculate_similarity_scores(self, name1: str, name2: str) -> Dict[str, float]:
        # base normalisée (latin pour les deux si besoin)
        n1 = self.normalize_name(name1)
        n2 = self.normalize_name(name2)

        # formes canoniques (latin canonisé)
        c1 = self._canonicalize_latin(n1)
        c2 = self._canonicalize_latin(n2)

        # scores de base
        jw = jellyfish.jaro_winkler_similarity(n1, n2)
        lev_d = jellyfish.levenshtein_distance(n1, n2)
        max_len = max(len(n1), len(n2))
        lev = 1 - (lev_d / max_len) if max_len > 0 else 0
        sdx = 1.0 if jellyfish.soundex(n1) == jellyfish.soundex(n2) else 0.0

        # scores sur canonique
        jw_c = jellyfish.jaro_winkler_similarity(c1, c2)
        lev_c_d = jellyfish.levenshtein_distance(c1, c2)
        max_len_c = max(len(c1), len(c2))
        lev_c = 1 - (lev_c_d / max_len_c) if max_len_c > 0 else 0

        # bonus phonétique (metaphone)
        m1 = self._phonetic_code(c1 or n1)
        m2 = self._phonetic_code(c2 or n2)
        phon = 1.0 if m1 == m2 and m1 != '' else 0.0

        # bonus Jaccard sur tokens
        t1, t2 = self._token_set(n1), self._token_set(n2)
        inter = len(t1 & t2)
        union = len(t1 | t2) if (t1 | t2) else 1
        jacc = inter / union

        # composite
        composite = (
            0.28 * jw +
            0.22 * lev +
            0.10 * sdx +
            0.22 * jw_c +
            0.10 * lev_c +
            0.06 * phon +
            0.02 * jacc
        )

        # infos debug utiles pour affichage
        return {
            'latin_input': n1,
            'latin_candidate': n2,
            'canonical_input': c1,
            'canonical_candidate': c2,
            'jaro_winkler': round(jw, 4),
            'levenshtein': round(lev, 4),
            'soundex': sdx,
            'jaro_winkler_canon': round(jw_c, 4),
            'levenshtein_canon': round(lev_c, 4),
            'phonetic': phon,
            'jaccard_tokens': round(jacc, 4),
            'composite': round(composite, 4)
        }

    # =========================
    # Recherche & Décision
    # =========================
    def find_best_match(self, input_name: str, threshold: float = 0.7) -> Dict:
        if self.df is None:
            return {"error": "Données non chargées"}
        best_match = None
        best_score = 0.0
        for idx, row in self.df.iterrows():
            sims = self.calculate_similarity_scores(input_name, row['Full Name'])
            comp = sims['composite']
            if comp > best_score:
                best_score = comp
                best_match = {
                    'index': idx,
                    'full_name': row['Full Name'],
                    'nationality': row.get('Nationality', 'Unknown'),
                    'risk_category': row.get('Risk Category', 'Unknown'),
                    'source_funds': row.get('Source of Funds', 'Unknown'),
                    'watchlist_db': row.get('Watchlist Database', 'Unknown'),
                    'notes': row.get('Notes', ''),
                    'similarity_scores': sims,
                    'composite_similarity': comp
                }
        return best_match if best_score >= threshold else None

    def make_decision(self, input_name: str) -> Dict:
        match = self.find_best_match(input_name)
        if not match:
            return {
                "decision": "ALLOWED",
                "confidence": 0.95,
                "reason": "Aucune correspondance trouvée dans la watchlist",
                "similarity": 0.0,
                "match_details": None
            }
        similarity_score = match['composite_similarity']
        risk_category = match['risk_category']
        risk_weight = self.risk_weights.get(risk_category, 0.5)
        final_score = similarity_score * (0.6 + risk_weight * 0.4)

        if final_score >= 0.85 and similarity_score >= 0.9:
            decision = "BLOCKED"
            confidence = min(final_score + 0.05, 1.0)
            reason = f"Très forte similarité ({similarity_score*100:.1f}%) avec {risk_category}"
        elif final_score >= 0.7 and similarity_score >= 0.8:
            decision = "REVIEW"
            confidence = final_score
            reason = f"Forte similarité ({similarity_score*100:.1f}%) avec {risk_category}"
        elif final_score >= 0.5:
            decision = "REVIEW"
            confidence = final_score * 0.8
            reason = f"Similarité modérée ({similarity_score*100:.1f}%) avec {risk_category}"
        else:
            decision = "ALLOWED"
            confidence = 1 - final_score
            reason = f"Similarité faible ({similarity_score*100:.1f}%)"
        return {
            "decision": decision,
            "confidence": round(confidence, 4),
            "reason": reason,
            "similarity": round(similarity_score * 100, 2),
            "match_details": match
        }

    # =========================
    # Sauvegarde / Chargement
    # =========================
    def save_models(self, models_dir: str = 'models'):
        import os
        os.makedirs(models_dir, exist_ok=True)
        if self.rf_model:
            with open(f'{models_dir}/rf_model.pkl', 'wb') as f:
                pickle.dump(self.rf_model, f)
        if self.xgb_model:
            self.xgb_model.save_model(f'{models_dir}/xgb_model.json')
        if self.lr_model:
            with open(f'{models_dir}/lr_model.pkl', 'wb') as f:
                pickle.dump(self.lr_model, f)
        with open(f'{models_dir}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        with open(f'{models_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(f'{models_dir}/feature_names.pkl', 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"Modèles sauvegardés dans {models_dir}/")

    def load_models(self, models_dir: str = 'models'):
        try:
            with open(f'{models_dir}/rf_model.pkl', 'rb') as f:
                self.rf_model = pickle.load(f)
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(f'{models_dir}/xgb_model.json')
            with open(f'{models_dir}/lr_model.pkl', 'rb') as f:
                self.lr_model = pickle.load(f)
            with open(f'{models_dir}/label_encoders.pkl', 'rb') as f:
                self.label_encoders = pickle.load(f)
            with open(f'{models_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open(f'{models_dir}/feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            print("Modèles chargés avec succès")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement des modèles: {e}")
            return False


def main():
    aml_system = AMLNameMatcher('cleaned_aml_data.xlsx')
    if not aml_system.load_and_prepare_data():
        print("Erreur lors du chargement des données")
        return
    if not aml_system.train_models(test_size=0.3, use_cross_validation=True):
        print("Erreur lors de l'entraînement")
        return
    aml_system.save_models()
    print("\n=== SYSTÈME AML PRÊT ===")
    print("Entrez 'quit' pour quitter")
    while True:
        name_input = input("\nEntrez le nom complet à vérifier: ").strip()
        if name_input.lower() == 'quit':
            break
        if not name_input:
            print("Veuillez entrer un nom valide")
            continue
        print("\n...Processing...")
        result = aml_system.make_decision(name_input)
        print(f"\nDecision: {result['decision']}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print(f"Reason: {result['reason']}")
        if result['match_details']:
            match = result['match_details']
            sims = match['similarity_scores']
            print(f"\nTop match found:")
            print(f"  Name: {match['full_name']}")
            print(f"  Similarity: {result['similarity']:.2f}%")
            print(f"  Risk Category: {match['risk_category']}")
            print(f"  Nationality: {match['nationality']}")
            print(f"  Notes: {match['notes']}")
            print(f"\nTranslittération & formes utilisées:")
            print(f"  latin_input        = {sims.get('latin_input','')}")
            print(f"  latin_candidate    = {sims.get('latin_candidate','')}")
            print(f"  canonical_input    = {sims.get('canonical_input','')}")
            print(f"  canonical_candidate= {sims.get('canonical_candidate','')}")
            print(f"\nSimilarity Details:")
            for k, v in sims.items():
                if k in ('latin_input','latin_candidate','canonical_input','canonical_candidate'):
                    continue
                if k in ('soundex','phonetic'):
                    print(f"  {k}: {'Match' if v == 1.0 else v}")
                else:
                    print(f"  {k}: {v}")
        else:
            print("  (aucune correspondance)")

if __name__ == "__main__":
    main()
