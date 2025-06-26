import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency, f_oneway

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

st.title('Social Survey Data Explorer: Categorical vs. Numerical')

# Load data
def load_data():
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
        'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
        'hours_per_week', 'native_country', 'income'
    ]
    df = pd.read_csv('adult.csv', header=None, names=columns, na_values='?')
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.dropna(inplace=True)
    return df

df = load_data()

cat_cols = [
    'workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income'
]
num_cols = [
    'age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week'
]

st.sidebar.header('Explore Relationships')
col1 = st.sidebar.selectbox('Select First Variable', cat_cols + num_cols)
col2 = st.sidebar.selectbox('Select Second Variable', cat_cols + num_cols)

# Helper: is categorical?
def is_cat(col):
    return col in cat_cols

def is_num(col):
    return col in num_cols

st.write(f'### Relationship: {col1} vs. {col2}')

# 1. Numerical vs. Numerical
if is_num(col1) and is_num(col2):
    st.write('#### Scatterplot')
    fig, ax = plt.subplots()
    sns.scatterplot(x=col1, y=col2, data=df, hue='income', alpha=0.5, ax=ax)
    st.pyplot(fig)
    st.write('#### Correlation Matrix')
    corr_val = df[[col1, col2]].corr().iloc[0,1]
    st.write(f'**Pearson correlation coefficient:** {corr_val:.2f}')
    fig2, ax2 = plt.subplots()
    sns.heatmap(df[[col1, col2]].corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)
    with st.expander('Key Insight'):
        if abs(corr_val) > 0.5:
            st.write(f"There is a strong {'positive' if corr_val > 0 else 'negative'} correlation between {col1} and {col2}.")
        elif abs(corr_val) > 0.2:
            st.write(f"There is a moderate correlation between {col1} and {col2}.")
        else:
            st.write(f"There is little to no linear correlation between {col1} and {col2}.")

# 2. Categorical vs. Numerical
elif (is_cat(col1) and is_num(col2)) or (is_num(col1) and is_cat(col2)):
    cat_col = col1 if is_cat(col1) else col2
    num_col = col2 if is_num(col2) else col1
    st.write('#### Boxplot')
    fig, ax = plt.subplots()
    sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.write('#### Violin Plot')
    fig2, ax2 = plt.subplots()
    sns.violinplot(x=cat_col, y=num_col, data=df, ax=ax2)
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    st.write('#### Barplot (Mean)')
    fig3, ax3 = plt.subplots()
    sns.barplot(x=cat_col, y=num_col, data=df, estimator=np.mean, ci=None, ax=ax3)
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    st.write('#### Group Summary Statistics')
    st.write(df.groupby(cat_col)[num_col].describe())
    # ANOVA
    groups = [group[num_col].values for name, group in df.groupby(cat_col)]
    if len(groups) > 1:
        fstat, pval = f_oneway(*groups)
        st.write(f'**ANOVA F-statistic:** {fstat:.2f}, **p-value:** {pval:.4f}')
        with st.expander('Key Insight'):
            if pval < 0.05:
                st.write(f"There is a statistically significant difference in {num_col} across groups of {cat_col}.")
            else:
                st.write(f"No significant difference in {num_col} across groups of {cat_col}.")

# 3. Categorical vs. Categorical
elif is_cat(col1) and is_cat(col2):
    st.write('#### Grouped Bar Chart')
    fig, ax = plt.subplots()
    cross = pd.crosstab(df[col1], df[col2], normalize='index')
    cross.plot(kind='bar', stacked=True, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    st.write('#### Crosstab Heatmap')
    fig2, ax2 = plt.subplots()
    sns.heatmap(pd.crosstab(df[col1], df[col2]), annot=True, fmt='d', cmap='Blues', ax=ax2)
    st.pyplot(fig2)
    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(pd.crosstab(df[col1], df[col2]))
    st.write(f'**Chi-square statistic:** {chi2:.2f}, **p-value:** {p:.4f}')
    with st.expander('Key Insight'):
        if p < 0.05:
            st.write(f"There is a statistically significant association between {col1} and {col2}.")
        else:
            st.write(f"No significant association between {col1} and {col2}.")

# 4. Highlight Key Insights
st.write('---')
st.write('### Example Insights')
st.markdown('''
- Males tend to work more hours per week than females across most education levels.
- Higher education correlates with higher income.
- Capital gain is skewed towards high-income individuals.
''') 