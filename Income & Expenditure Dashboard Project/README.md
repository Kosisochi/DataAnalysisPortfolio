# Data Visualization Project

## Skills Required
- PowerBI
- DAX

#  Project Title: Income & Expenditure Dashboard 
Edith is 30 y/o woman sharing an apartment with a flatmate. With the increasing cost of living and inflation, Edith wants to keep track of her income and expenses, identigy where she needs to reduce spending or where she needs a different strategu to help her save more. 
She also need sto see how her income has fluctuated over the years as the cost of living increases. This will help her negotiate for an increase in salary or find an additional source of income. 

# Project Goal:
The goal of this dashboard is for Edith to monitor her income, savings and expenditure and quickly answer questions such as 
1. How much have I earned, spent or saved in a specific time period
2. On what category do I spend the most?
3. What percentage of my income am I saving?
4. What is my credit utilization rate and what has it historically being?
5. Are my utility bills increasing at an alarming rate?
   

# Data:
**Data Source**: The data was created from pdf file of a bank statement. The entries were inputted into an Excel file, anonymized,  normalized andloaded into Poer BI

**Data Description**: It contains 463 rows and 7 columns. The column headers are : ID (primary key), Date, Item, Amount, TT_ID (secondary key), AT_ID (secondary key), CAT_ID (secondary key)

**Data Cleaning**: Data Anoymization was done on the data to remove any connection to living person(s) especially in the Item description columns.

**Data Quality Checks**: Rows are complete, all datatypes are correct. 

**Data Model**
The data model is a simple star schema. The data was normalized to create the Dimension Tables (DIM_Accountype, DIM_TransactionType, DIM_Category) and the Fact Table (FACT_Records)
 ![Data Model](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Income%20%26%20Expenditure%20Dashboard%20Project/Data%20Model.PNG)

# Dashboard

[**PBIX PROJECT FILE**](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Income%20%26%20Expenditure%20Dashboard%20Project/Income%20%26%20Expenditure%20Dashboard.pbix)


### This dashboard is divided into 4 pages
**Pages 1**: Shows the overview page which displays high level overview of the rest of the dashboard. From this page, the user can drill to other analysis and detail pages.
  ![OverviewPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Income%20%26%20Expenditure%20Dashboard%20Project/Overview%20Page.PNG)

**Pages 2**: This is first analysis page. It focuses on the Credit Card Account. 
![CreditCardPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Income%20%26%20Expenditure%20Dashboard%20Project/Credit%20Card%20Page.PNG)

**Page 3**: This is the second analysis page: It focuses on the Debit Card Account.
![DebitCardPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Income%20%26%20Expenditure%20Dashboard%20Project/Debit%20Card%20Page.PNG)

**Page 4**: This is the detail page. The user drills through to this page when they need fine grain level details
![DetailPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Income%20%26%20Expenditure%20Dashboard%20Project/Details%20Page.PNG)


# Measures and Columns
**Total Income**
```sql
Total Income = CALCULATE(SUM(FACT_Records[Amount]), 
FACT_Records[Cat_ID] IN {"CID3","CID10", "CID13", "CID16",  "CID18", "CID19"} 
)
```

**Total Expenditure**
```sql
Total Expenditure = CALCULATE(SUM(FACT_Records[Amount]),
 FACT_Records[Cat_ID] IN
{"CID1", "CID2", "CID4", "CID5", "CID6", "CID7", "CID8", "CID9", "CID11", "CID12", "CID14", "CID15", "CID17", "CID20"}
)
```

**Net Cash Flow**
```sql
Net Cash Flow = [Total Income] - [Total Expenditure]
**Total Savings**
```sql
Total Savings = CALCULATE(SUM(FACT_Records[Amount]), 
FILTER(FACT_Records, FACT_Records[Cat_ID] = "CID18")
)
```

**Savings Target**
```sql
Savings Target = 0.2 * [Total Income]
```

**Percentage of Income Saved**
```sql
Percentage of Income Saved = DIVIDE([Total Savings],[Total Income] ,0)
```

**Credit Utilization Rate**
```sql
Credit Utilization Rate = 
VAR CreditLimit = 4000
VAR TotalSpent = 
CALCULATE(SUM(FACT_Records[Amount]), 
FACT_Records[AT_ID] = "AT2",
FACT_Records[TT_ID] = "TT2",
FACT_Records[Cat_ID] <> "CID13"
)
RETURN
DIVIDE (TotalSpent,CreditLimit,0)
```


# Findings and Recommendations
1. Edith has vaved only 2.9% of her income. This is considered extremely low. She needs to look into more saving strategies.
2. Overall, she is spending less than she earns with a positive nwt cash flow of about $8000. 
3. Her major expenses go to her rent, she has paid $15,300 from April 2024 to December 2024. Compared to her income her rent is high at 38.8% of income (transferred funds + salary).
4. Most of her expenses are coming from her Chequing Account and her Credit Utilizatio Rate (CUR) has not exceeded 14%.
5. Her main source of income over the specified time is from funds transferrred ($24,623) from her other accounts to this account, followed by her total salary ($14,774).


# Difficulties Encountered and Future Work
The data came in the form of pdf bank statements and had to be inputted manually into an Excel file. This was a time conusuming task and the time spent will increase as more months pass by generating a pdf for each one. Since bank statements contain a lot of sensitive information, using a third party pdf data extractor was not a good idea . Future work will include building a Gen-AI based pdf extarctor that runs on your local machine. 

**FYI**
This data has been anonymised to remove an identifying information connected to a living person(s).

