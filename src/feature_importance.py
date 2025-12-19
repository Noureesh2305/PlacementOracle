import pandas as pd
import matplotlib.pyplot as plt

def show_feature_importance(model, feature_names):
    importances = model.feature_importances_
    features = pd.Series(importances, index=feature_names)
    features.sort_values(ascending=False).head(10).plot(kind='bar')
    plt.title("Top 10 Important Features")
    plt.show()
