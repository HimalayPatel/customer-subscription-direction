import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser

customer_data=pd.read_csv("C:\\Users\\himal\Desktop\\Machine Learning Practicals\\P3- Directing Customers to Subscription through App Behavior Analysis\\P39-CS3-Data\\appdata10.csv")

#print(customer_data.head())
#print(customer_data.describe())
customer_data['hour']=(customer_data['hour'].str.slice(1,3)).astype(int)
#print(customer_data['hour'])
customer_data2=customer_data.drop(columns=['user','screen_list','enrolled_date','first_open','enrolled'])
#print(customer_data2.head())
#print(customer_data2.describe())

# Histogram
plt.suptitle('Histograms of Numerical Columns',fontsize=20)
for i in range(1,customer_data2.shape[1]+1):
    plt.subplot(3,3,i)
    axes=plt.gca()
    axes.set_title(customer_data2.columns.values[i-1])
    vals=np.size(customer_data2.iloc[:,i-1].unique())
    plt.hist(customer_data2.iloc[:,i-1],bins=vals,color='#FF5500')
plt.show()

# Correlation with Response Variable
customer_data2.corrwith(customer_data.enrolled).plot.bar(figsize=(15,10),
                        title='Correlation with Response Variable',
                        fontsize=10, rot=45,
                        grid=True)
plt.show()

# Correlation matrix
sns.set(style="white", font_scale=2)
corr = customer_data2.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle("Correlation Matrix", fontsize = 40)
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
plt.show()

# Feature Engineering- Response Variable
#print(customer_data.dtypes)
customer_data["first_open"] = [parser.parse(row_date) for row_date in customer_data["first_open"]]
customer_data["enrolled_date"] = [parser.parse(row_date) if isinstance(row_date, str) else row_date for row_date in customer_data["enrolled_date"]]
#print(customer_data.dtypes)

# Selecting Time For Response
customer_data["difference"] = (customer_data.enrolled_date-customer_data.first_open).astype('timedelta64[h]')
# response_hist = plt.hist(customer_data["difference"].dropna(), color='#3F5D7D')
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()

plt.hist(customer_data["difference"].dropna(), color='#3F5D7D', range = [0, 100])
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()

plt.hist(customer_data["difference"].dropna(), color='#3F5D7D', range = [0, 48])
plt.title('Distribution of Time-Since-Screen-Reached')
plt.show()

customer_data.loc[customer_data.difference > 48, 'enrolled'] = 0
customer_data = customer_data.drop(columns=['enrolled_date', 'difference', 'first_open'])

# Feature Engineering- Screens
top_screens = pd.read_csv("C:\\Users\\himal\Desktop\\Machine Learning Practicals\\P3- Directing Customers to Subscription through App Behavior Analysis\\P39-CS3-Data\\top_screens.csv").top_screens.values
customer_data["screen_list"] = customer_data.screen_list.astype(str) + ','
for sc in top_screens:
    customer_data[sc] = customer_data.screen_list.str.contains(sc).astype(int)
    customer_data['screen_list'] = customer_data.screen_list.str.replace(sc+",", "")
customer_data['Other'] = customer_data.screen_list.str.count(",")
customer_data = customer_data.drop(columns=['screen_list'])

# Funnels
savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]
customer_data["SavingCount"] = customer_data[savings_screens].sum(axis=1)
customer_data = customer_data.drop(columns=savings_screens)

cm_screens = ["Credit1",
               "Credit2",
               "Credit3",
               "Credit3Container",
               "Credit3Dashboard"]
customer_data["CMCount"] = customer_data[cm_screens].sum(axis=1)
customer_data = customer_data.drop(columns=cm_screens)

cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
customer_data["CCCount"] = customer_data[cc_screens].sum(axis=1)
customer_data = customer_data.drop(columns=cc_screens)

loan_screens = ["Loan",
               "Loan2",
               "Loan3",
               "Loan4"]
customer_data["LoansCount"] = customer_data[loan_screens].sum(axis=1)
customer_data = customer_data.drop(columns=loan_screens)

# print(customer_data.head())
# print(customer_data.describe())
# print(customer_data.columns)

customer_data.to_csv('new_appdata10.csv', index = False)
