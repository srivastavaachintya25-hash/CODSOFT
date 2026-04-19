from sklearn.preprocessing import StandardScaler


def load_data():
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/creditcard.csv")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    from sklearn.preprocessing import StandardScaler

    return train_test_split(X, y, test_size=0.2, random_state=42)