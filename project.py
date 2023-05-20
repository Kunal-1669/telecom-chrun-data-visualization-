#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%%
df1=pd.read_csv("cell2celltrain.csv")
#df2=pd.read_csv("cell2cellholdout.csv")

# %%
df1.info()
#%%
df1 = df1.set_index('CustomerID')
#%%
num_cols1 = df1.select_dtypes(include='number').columns
print(f"Number of numerical columns: {len(num_cols1)}")

# Get the categorical columns
cat_cols1 = df1.select_dtypes(include='object').columns
print(f"Number of categorical columns: {len(cat_cols1)}")
# %%
df1.isna().sum()
#%%
null_counts = df1.isnull().sum()

# Identify columns with null values
cols_with_nulls = null_counts[null_counts > 0].index.tolist()
#%%
# Print the column names and their respective null value counts
for col in cols_with_nulls:
    print(f"Column '{col}' has {null_counts[col]} null values.")
#%%
df1.fillna(df1.mean(),inplace=True)
#%%
df1['ServiceArea'].fillna(df1['ServiceArea'].mode()[0],inplace=True)
#%%
# sns.boxplot(df1['AgeHH1'])
# %%

Q1 = df1.quantile(0.25)
Q3 = df1.quantile(0.75)
IQR = Q3 - Q1
#%%
print("First quartile")
print(Q1)
#%%
print("THird quartile")
print(Q3)
#%%
print("IQR")
print(IQR)
#%%
# Remove outliers
df1 = df1[~((df1 < (Q1 - 1.5 * IQR)) |(df1 > (Q3 + 1.5 * IQR))).any(axis=1)]
# %%
df1.shape
#%%
df1.to_csv('/Users/kunal/Documents/dataviz/cleandata.csv', index=False)
#%%

df2=pd.read_csv("cleandata.csv")
#%%
# %%
numeric_cols = df1.select_dtypes(include=['number']).columns.tolist()
# df_numeric = df1[numeric_cols]
#%%
cat_cols = df1.select_dtypes(include=['object']).columns.tolist()
df_cat = df1[cat_cols]
#%%
# numeric_cols.remove('CallForwardingCalls','RetentionCalls','RetentionOffersAccepted','ReferralsMadeBySubscriber','AdjustmentsToCreditRating')
numeric_cols.remove('CallForwardingCalls')
numeric_cols.remove('RetentionCalls')
numeric_cols.remove('RetentionOffersAccepted')
numeric_cols.remove('ReferralsMadeBySubscriber')
numeric_cols.remove('AdjustmentsToCreditRating')

# %%
from sklearn.preprocessing import StandardScaler

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df1[numeric_cols])
import numpy as np

# Compute the covariance matrix
cov_matrix = np.cov(X_scaled.T)
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
# Sort the eigenvalues in descending order
sorted_idx = np.argsort(eigen_values)[::-1]
sorted_eigenvalues = eigen_values[sorted_idx]

# Compute the explained variance ratio
explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)
# Compute the cumulative sum of the explained variance ratio
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Find the number of principal components that capture 95% of the variance
n_components = np.argmax(cumulative_variance_ratio >= 0.95) + 1
from sklearn.decomposition import PCA

# Transform the data into the new lower-dimensional space
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)
# Check the condition number and singular values
condition_number = np.linalg.cond(X_pca)
singular_values = np.linalg.svd(X_pca, compute_uv=False)

print(f"Condition number: {condition_number}")
print(f"Singular values: {singular_values}")


# %%
n_components
#%%
import matplotlib.pyplot as plt

# Get the explained variance ratio and the cumulative sum
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio) * 100

# Get the number of components for 95% explained variance
n_components = np.argmax(cumulative_variance_ratio >= 95) + 1

# Plot the cumulative explained variance ratio
plt.plot(range(1, len(cumulative_variance_ratio)+1), cumulative_variance_ratio, marker='o')

# Plot the dashed lines for 95% explained variance and the optimum number of features
plt.axhline(y=95, color='black', linestyle='--')
plt.axvline(x=n_components, color='red', linestyle='--')

# Add labels and title
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance (%)')
plt.xticks(range(1, len(cumulative_variance_ratio)+1), range(1, len(cumulative_variance_ratio)+1))

# add labels and title
plt.title('Cumulative Explained Variance vs Number of Components')

plt.show()

#%%%
df_num=df1[numeric_cols]

# %%

#%%

#%%
from scipy.stats import shapiro

stat, p = shapiro(df1[numeric_cols])


alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')
#%%
log_data = np.log(df1[numeric_cols])
#%%
from scipy.stats import shapiro


stat, p = shapiro(log_data)


alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')
# df1.shape
#%%


# %%
df1.shape
#%%
df1.drop(['CallForwardingCalls','RetentionCalls','RetentionOffersAccepted','ReferralsMadeBySubscriber'],axis=1,inplace=True)
# %%
corr_matrix = df1.corr()
#%%%
# corr_with_y = corr_matrix['Churn']
# corr_with_y = corr_with_y.drop(['Churn'], axis=0) # remove the correlation of y with itself

# # %%
# sns.barplot(x=corr_matrix.values, y=corr_matrix.index, orient='h')
# plt.show()

# %%
corr_matrix
# %%

sns.set(style='white')
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, annot=True, fmt='.2f', linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
# %%
datacorr = df1[["MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge", "CustomerCareCalls", "ThreewayCalls", "ReceivedCalls", "OutboundCalls", "MonthsInService", "HandsetPrice", "CreditRating"]]


# %%
corr_matrix = datacorr.corr()
# %%
sns.heatmap(corr_matrix, annot=True,fmt='.2f', annot_kws={"fontsize":12}, cbar_kws={'label': 'Pearson correlation coefficient'}, square=True)

plt.title('Pearson correlation coefficient matrix', fontsize=16)
plt.show()
df_churn=df1[df1['Churn']=='Yes']
# %%
df_retain=df1[df1['Churn']=='No']
#%%
sns.countplot(x='Churn',data=df1)
plt.show()
#%%

sns.histplot(data=df_churn, x="MonthsInService", hue="Churn", element="step", alpha=0.6, color='aquamarine4', fill=True, palette="pastel")
sns.set_style("whitegrid")
sns.set_palette("Set1")
sns.set(rc={'figure.figsize':(8,6)})
sns.set(font_scale=1.2)
sns.set_style({"axes.facecolor": "0.97", "axes.edgecolor": "0.65", "grid.color": "0.85"})
sns.despine(left=True)
sns.set_palette("pastel")
plt.xlabel("Service period for churned customers (In Months)")
plt.ylabel("Frequency")
plt.title("Service Months Distribution for Churned customers")
plt.show()
# %%
import seaborn as sns

sns.lineplot(data=df1, x='MonthsInService', y='MonthlyMinutes',hue='Churn')
plt.show()
# %%
import seaborn as sns

sns.set(style="whitegrid", palette="muted")

sns.lmplot(x='OverageMinutes', y='MonthlyRevenue', data=df1, hue='Churn', col='Churn', scatter_kws={'alpha':0.4})
sns.despine(left=True)
plt.show()
# %%
import seaborn as sns

sns.set(style="darkgrid")
sns.lmplot(x="OverageMinutes", y="MonthlyRevenue", data=df1, ci=None)
sns.set(rc={'figure.figsize':(10,6)})
plt.xlabel("Overage minutes used by the customer")
plt.ylabel("Monthly revenue of the Telecom company")
plt.show()
#%%
sns.countplot(data=df1, x='PrizmCode', hue='Churn', palette='Set2', dodge=True)
plt.title('Churn distribution for Prizm code')
plt.show()

# %%

# %%
import plotly.express as px
fig = px.pie(df_churn, names='PrizmCode')
fig.show()
#%%
import seaborn as sns

sns.displot(data=df1, x='AgeHH1', bins=10, kde=True,hue="Churn")

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution of Customers')

# Show plot
plt.show()
#%%
import seaborn as sns

sns.boxplot(data=df1, y='AgeHH1', x='Churn')
sns.set_style('whitegrid')
sns.despine(left=True)
sns.set(rc={'figure.figsize':(10,8)})
sns.set_context('talk')
sns.set_palette('colorblind')
sns.set(font_scale=1.5)
plt.title('Age of Customers Who Churned vs. Those Who Did Not')
plt.ylabel('Age')
plt.xlabel('Churn')
plt.show()
#%%
sns.barplot(data=df1, x="CreditRating", y="MonthlyRevenue", hue="Churn", estimator=sum)
plt.show()
# Plot boxplot
plt.hist(df1['CreditRating'])

# Add labels and title
plt.xlabel('Credit Ratings')
plt.ylabel('Score')
plt.title('Credit Ratings of Customers')

# Show plot
plt.show()
#%%
import seaborn as sns

sns.set_style("whitegrid")

sns.catplot(
    data=df1,
    x="OptOutMailings",
    hue="Churn",
    kind="count",
    height=6,
    aspect=1.5,
    palette="Set1",
    alpha=.8
).set(
    xlabel="Opt Out Mailings",
    ylabel="Count",
    title="Stacked Bar Plot of OptOutMailings and Churn"
)
plt.show()
sns.violinplot(x='Churn', y='MonthlyRevenue', data=df1)

# Add labels and title
plt.xlabel('Churn')
plt.ylabel('Monthly Revenue')
plt.title('Account Length Distribution for Churned and Non-Churned Customers')

# Show the plot
plt.show()
import seaborn as sns

# Select only numeric columns
num_cols = ['MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge', 'DirectorAssistedCalls',
]

# Create pair plot
sns.pairplot(data=df1[num_cols])
plt.show()
#%%
fig = px.scatter(df_churn, x="OverageMinutes", y="MonthlyRevenue", trendline="ols", color_discrete_sequence=["brown"])
fig.update_layout(
xaxis_title="Overage minutes used by the customer",
yaxis_title="Monthly revenue of the Telecom company",
template="plotly_white"
)
fig.show()

# %%
fig = px.scatter(df_retain, x="OverageMinutes", y="MonthlyRevenue", trendline="lowess",
labels={"OverageMinutes": "Overage minutes used by the customer",
"MonthlyRevenue": "Monthly revenue of the Telecom company "},
color_discrete_sequence=["darkgreen"])
fig.update_layout(title="Relationship between Overage minutes and Monthly revenue",
xaxis_title="Overage minutes used by the customer",
yaxis_title="Monthly revenue of the Telecom company ",
plot_bgcolor='white')
fig.show()
#%%
sns.kdeplot(data=df1, x='TotalRecurringCharge', hue='Churn',fill=True)
plt.xlabel('Total Recurring Charge')
plt.ylabel('Density')
plt.title('Total Recurring Charge KDE ')
plt.show()
# %%
sns.kdeplot(data=df1, x='TotalRecurringCharge', hue='PrizmCode')
plt.xlabel('Total Recurring Charge')
plt.ylabel('Density')
plt.title('Total Recurring Charge KDE ')
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# Set figure size and style
plt.figure(figsize=(15, 10))
sns.set_style("whitegrid")
# plt.xticks(rotation=90)

# Define the list of columns to plot
columns = ['Occupation', 'IncomeGroup', 'CreditRating', 'NonUSTravel', 'PrizmCode', 'MaritalStatus']

# Loop through each column and create a subplot
for i, column in enumerate(columns):
    plt.subplot(2, 3, i+1)
    sns.countplot(x=column, hue='Churn', data=df1)
    plt.title(column)
    plt.xticks(rotation=90)

# Set the title of the entire plot
plt.suptitle("Churn with Respect to Occupation, Income Group, Credit Rating, Non-US Travel, Prizm Code, and Marital Status")
plt.tight_layout()
# Show the plot
plt.show()
# %%

# %%
df1.describe()
# %%
df1.head()
# %%

# %%
