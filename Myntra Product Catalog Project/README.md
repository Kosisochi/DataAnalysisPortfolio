# Data Visualization Project

## This project is about building a dashboard in PowerBI.

## Skills Required
- PowerBI, DAX

#  Project Title: Myntra Product Catalog. Myntra is an e-commerce platform originating from India. It sells clothing itesm from multiple curated brands. It services woem, men, girls, boys and usiex for bith adults and kids. 

# Data:
Data Source: The data was download from Kaggle [Add link Here]
Data Description: it contains .... rows and ... colums. The colums headers are :  ......

Data Cleaning: No data cleaning step was required because the data was already in the format required. 
Data Quality Checks: Rows are completed, all datatype are corrected. The color colums has some missing values which was replaced wtih text "Not SPecified"


Project Goal: The goal of this dashboard is for the Myntra Product Catalog manger to keep an eye on thier current current catalog and be able to easilty answer questions such as 1. How many product does Myntra currently have? 2. Which gender does their current catalog cater to. 3. How many brands do they have stocked. Which brand has trhe mopst expensive product? Which brand has the highest total sales value? 

This will help decide if they want to add more brands or reduce?


# Analysis

#### This dashboard is divided into 4 pages
* Pages 1: Shows the summary page which displays high level overview of the rest of the dashboar. From this page, the user can drill to other analysis and detail pages.


  ![SummaryPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Summary%20Page.PNG)

  ```DAX
DistinctProduct = DISTINCTCOUNT(myntra_products_catalog[ProductID])
  ```

* Pages 2: This is first analysis page. It focuses on the Product ANaysis. 

* Page 3: This is the second analysis page: It focuses on the Color Analysis.




*FYI: the anlysis on in the project was restricted by what was available in the data. No attempt was made to source for additional external data. 
