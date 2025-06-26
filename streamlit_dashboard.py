import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Social Survey Data Dashboard", layout="wide")

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Load data
def load_data():
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
        'hours_per_week', 'native_country', 'income'
    ]
    df = pd.read_csv('adult.csv', header=None, names=columns, na_values='?')
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.dropna(inplace=True)
    return df

df = load_data()
cat_cols = [
    'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income'
]
num_cols = [
    'age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'
]

# Sidebar menu
st.sidebar.title("Menu")
page = st.sidebar.radio("Go to", [
    "Home / Overview",
    "Dataset Preview",
    "Categorical Data Analysis",
    "Numerical Data Analysis",
    "Explore Relationships",
    "Sampling Techniques",
    "ML Model Comparison",
    "Conclusion"
])

# --- Home / Overview ---
if page == "Home / Overview":
    st.markdown("# 🏠 Social Survey Data Dashboard")
    st.markdown("""
    Welcome! This dashboard provides a comparative analysis of **categorical** and **numerical** data from a social survey (UCI Adult dataset).\
    Use the sidebar to navigate through data previews, visualizations, sampling, and machine learning comparisons.\
    
    **Goal:** To help students understand how different data types are analyzed and compared in statistics and ML.
    """)

# --- Dataset Preview ---
elif page == "Dataset Preview":
    st.markdown("# 📋 Dataset Preview")
    st.write("Below is a preview of the first 10 rows of the dataset:")
    st.dataframe(df.head(10))
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Columns:**", ', '.join(df.columns))

# --- Categorical Data Analysis ---
elif page == "Categorical Data Analysis":
    st.markdown("# 🟦 Categorical Data Analysis")
    cat_col = st.selectbox("Select a categorical column", cat_cols)
    st.write(f"### Value Counts for {cat_col}")
    st.write(df[cat_col].value_counts())
    fig, ax = plt.subplots()
    df[cat_col].value_counts().plot(kind='bar', ax=ax)
    ax.set_ylabel('Count')
    ax.set_xlabel(cat_col)
    ax.set_title(f'{cat_col} Distribution')
    st.pyplot(fig)
    st.markdown(f"- **{cat_col}** is a categorical variable. Bar plots and value counts help us understand its distribution.")

# --- Numerical Data Analysis ---
elif page == "Numerical Data Analysis":
    st.markdown("# 🟩 Numerical Data Analysis")
    num_col = st.selectbox("Select a numerical column", num_cols)
    st.write(f"### Summary Statistics for {num_col}")
    st.write(df[num_col].describe())
    fig, ax = plt.subplots()
    df[num_col].plot(kind='hist', bins=20, ax=ax)
    ax.set_xlabel(num_col)
    ax.set_title(f'Histogram of {num_col}')
    st.pyplot(fig)
    st.markdown(f"- **{num_col}** is a numerical variable. Histograms show its distribution and spread.")

# --- Explore Relationships ---
elif page == "Explore Relationships":
    st.markdown("# 🔍 Explore Relationships")
    col1 = st.selectbox('Select First Variable', cat_cols + num_cols, key='rel1')
    col2 = st.selectbox('Select Second Variable', cat_cols + num_cols, key='rel2')
    def is_cat(col):
        return col in cat_cols
    def is_num(col):
        return col in num_cols
    if is_num(col1) and is_num(col2):
        st.write('## Scatterplot & Correlation')
        fig, ax = plt.subplots()
        sns.scatterplot(x=col1, y=col2, data=df, hue='income', alpha=0.5, ax=ax)
        st.pyplot(fig)
        corr_val = df[[col1, col2]].corr().iloc[0,1]
        st.write(f'**Pearson correlation coefficient:** {corr_val:.2f}')
        st.markdown(f"- Shows the linear relationship between **{col1}** and **{col2}**.")
    elif (is_cat(col1) and is_num(col2)) or (is_num(col1) and is_cat(col2)):
        cat_col = col1 if is_cat(col1) else col2
        num_col = col2 if is_num(col2) else col1
        st.write('## Boxplot & Group Stats')
        fig, ax = plt.subplots()
        sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.write(df.groupby(cat_col)[num_col].describe())
        groups = [group[num_col].values for name, group in df.groupby(cat_col)]
        if len(groups) > 1:
            fstat, pval = f_oneway(*groups)
            st.write(f'**ANOVA F-statistic:** {fstat:.2f}, **p-value:** {pval:.4f}')
        st.markdown(f"- Compares **{num_col}** across categories of **{cat_col}**.")
    elif is_cat(col1) and is_cat(col2):
        st.write('## Grouped Bar Chart & Crosstab')
        fig, ax = plt.subplots()
        cross = pd.crosstab(df[col1], df[col2], normalize='index')
        cross.plot(kind='bar', stacked=True, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        chi2, p, dof, expected = chi2_contingency(pd.crosstab(df[col1], df[col2]))
        st.write(f'**Chi-square statistic:** {chi2:.2f}, **p-value:** {p:.4f}')
        st.markdown(f"- Shows association between **{col1}** and **{col2}**.")

# --- Sampling Techniques ---
elif page == "Sampling Techniques":
    st.markdown("# 🧪 Sampling Techniques")
    st.write("## Simple Random Sampling")
    st.write(df.sample(5, random_state=1))
    st.write("## Stratified Sampling (by sex)")
    strat = df.groupby('sex', group_keys=False).apply(lambda x: x.sample(2, random_state=1))
    st.write(strat)
    st.write("## Cluster Sampling (by occupation)")
    clusters = np.random.choice(df['occupation'].unique(), 2, replace=False)
    st.write(df[df['occupation'].isin(clusters)].head(5))
    st.markdown("- Sampling methods help ensure representative data for analysis.")

# --- ML Model Comparison ---
elif page == "ML Model Comparison":
    st.markdown("# 🤖 ML Model Comparison")
    st.write("We compare models using only numerical, only categorical, and all features.")
    # Prepare data
    X_num = df[num_cols]
    X_cat = pd.get_dummies(df[cat_cols].drop('income', axis=1))
    y = LabelEncoder().fit_transform(df['income'])
    X_all = pd.concat([X_num, X_cat], axis=1)
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_num, y, test_size=0.2, random_state=42)
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cat, y, test_size=0.2, random_state=42)
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
    def eval_model(X_train, X_test, y_train, y_test):
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)
    st.write(f"Numerical only accuracy: {eval_model(Xn_train, Xn_test, yn_train, yn_test):.2f}")
    st.write(f"Categorical only accuracy: {eval_model(Xc_train, Xc_test, yc_train, yc_test):.2f}")
    st.write(f"All features accuracy: {eval_model(Xa_train, Xa_test, ya_train, ya_test):.2f}")
    st.markdown("- Combining both data types usually gives the best results.")

# --- Conclusion ---
elif page == "Conclusion":
    st.markdown("# 🏁 Conclusion")
    st.markdown("""
    - Categorical and numerical data require different analysis and visualization techniques.
    - Both types are important for understanding and predicting social survey outcomes.
    - Sampling and ML models benefit from a thoughtful approach to data types.
    
    **Thank you for exploring!**
    """) 