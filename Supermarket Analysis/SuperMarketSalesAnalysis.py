
"""
Supermarket to answer the following questions:
Which are the Top product_line by Total sales?
Which are the most selling product_line?
Which are the most preferred payment mode?
The most preferred means of payment for Each gender.
Which Gender spend more cash on product_line.
Top rated product_line.
"""

import pandas as pd
import numpy as np

# load in the data from a csv file
data = pd.read_csv("supermarket_sales.csv", sep=",")

# peep at the first 5 rows to see the column titles
print(data.head())

# inspect the data types of all the columns to see if its expected
print(data.dtypes)

# to get the accurate column names without case errors and so on
print(list(data.columns))


# convert the wrong data types to the correct one
data[['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']] \
    = data[['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']].astype("string")

# the Date column column contains different formats 00/00/0000 and 00-00-0000
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# i would like to split  this into specific columns
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

# print("####################")
# print(data.dtypes)

'''
Which are the Top product_line by Total sales?
'''
# Total Sales = (Unit Price * Quantity) + Tax 5%
# we already have this provided in Total column, so we don't need to calculate it

# to get the top sales, group Total column by Product line and then sort in descending order
data2 = pd.DataFrame(data.groupby('Product line')['Total'].sum())
TopProduct_line_by_Total_sales = data2.sort_values(by='Total', ascending=False)
print("The top product lines by total sales are : ", TopProduct_line_by_Total_sales)


'''
Which are the most selling product_line?
'''
# this is determined by the quantity of each product line sold
# Assumption made: The Quantity column refers to Quantity Sold and not Quantity Available/Produced

# to get the top selling product line, group Quantity column by Product line and then sort in descending order
data2 = pd.DataFrame(data.groupby('Product line')['Quantity'].sum())
TopProduct_line_by_QuantitySold = data2.sort_values(by='Quantity', ascending=False)
print("The top product line by quantity sold are : ", TopProduct_line_by_QuantitySold)


'''
Which are the most preferred payment mode?
'''
# count the unique occurrences from the Payment column

Preferred_Payment = data['Payment'].value_counts(sort=True)
print("The Preferred Payment Methods : ", Preferred_Payment.to_dict())
print("The Most Preferred Payment Methods : ", list(Preferred_Payment.to_dict().items())[0])


'''
The most preferred means of payment for Each gender.
'''
# group Preferred_Payment by Gender
# for each gender, redo the above preferred payment process

# subset the dataframe where Gender == Male
Male_subset = data.loc[data['Gender'] == "Male"]
Preferred_Payment = Male_subset['Payment'].value_counts(sort=True)
print("The Preferred Payment Methods for Male : ", Preferred_Payment.to_dict())
print("The Most Preferred Payment Methods for Male : ", list(Preferred_Payment.to_dict().items())[0])

# subset the dataframe where Gender == Female
Female_subset = data.loc[data['Gender'] == "Female"]
Preferred_Payment = Female_subset['Payment'].value_counts(sort=True)
print("The Preferred Payment Methods for Female : ", Preferred_Payment.to_dict())
print("The Most Preferred Payment Methods for Female : ", list(Preferred_Payment.to_dict().items())[0])


'''
Which Gender spends the most amount on each product_line?
'''
# for each of the product line show the top spending gender

# subset the dataframe where Gender == Male
Male_subset = data.loc[data['Gender'] == "Male"]
Product_line_by_Total_sales_Males = pd.DataFrame(Male_subset.groupby("Product line")['Total'].sum())
# renaming this so we can merge with the female df
Product_line_by_Total_sales_Males.rename(columns={'Total': 'Male'}, inplace=True)


# subset the dataframe where Gender == Female
Female_subset = data.loc[data['Gender'] == "Female"]
Product_line_by_Total_sales_Females = pd.DataFrame(Female_subset.groupby("Product line")['Total'].sum())
# renaming this so we can merge with the male df
Product_line_by_Total_sales_Females.rename(columns={'Total': 'Female'}, inplace=True)

# merge the male and female aggregated tables
PL_by_TS_Male_Female = Product_line_by_Total_sales_Males.merge(Product_line_by_Total_sales_Females, on='Product line')

# compare the contents of two columns based on multiple conditions and assigned values to a new column

conditions = ((PL_by_TS_Male_Female['Male'] > PL_by_TS_Male_Female['Female']),
              (PL_by_TS_Male_Female['Male'] < PL_by_TS_Male_Female['Female']),
              (PL_by_TS_Male_Female['Male'] == PL_by_TS_Male_Female['Female']))
choices = ['Male', 'Female', 'Both']
PL_by_TS_Male_Female['Top Spending Gender'] = np.select(conditions, choices, default=np.nan)
print('Top Spending Gender for each Product line : ', PL_by_TS_Male_Female)


'''
Which Gender spends the highest amount in total?
'''
print("Total_Amount_Spent_By_Males", Male_subset['Total'].sum())
print("Total_Amount_Spent_By_Females", Female_subset['Total'].sum())

if Male_subset['Total'].sum() > Female_subset['Total'].sum():
    print("Males spent the highest total amount : {}".format(Male_subset['Total'].sum()))
else:
    print("Females spent the highest total amount: {}".format(Female_subset['Total'].sum()))


'''
Top rated product_line.
'''
# aggregate the product line via their rating using groupby
TopRated = pd.DataFrame(data.groupby('Product line')['Rating'].sum())

# round up to a whole number
TopRated['Rating'] = TopRated['Rating'].round(decimals=0)

# sort
TopRated = TopRated.sort_values(by='Rating', ascending=False)

# convert the index to a column
TopRated = TopRated.reset_index()

# show result
print('Top rated product is {} with {} total ratings: '.format(TopRated['Product line'][0], TopRated['Rating'][0]))
