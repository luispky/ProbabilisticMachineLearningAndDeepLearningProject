#!/usr/bin/env python3

# packages |=======================================================
from sklearn.ensemble import RandomForestClassifier


# classifer |======================================================

# random forest classifier instance

rf_classifier = RandomForestClassifier(n_estimators = 200, min_samples_split = 500, max_depth = 20)