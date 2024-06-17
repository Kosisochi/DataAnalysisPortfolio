# Data Visualization Project

## This project is about building a dashboard in PowerBI.

## Skills Required
- PowerBI, DAX

#  Project Title: Myntra Product Catalog. Myntra is an e-commerce platform originating from India. It sells clothing itesm from multiple curated brands. It services woem, men, girls, boys and usiex for bith adults and kids. 

# Data:
Data Source: The data was download from Kaggle [Add link Here]
Data Description: it contains .... rows and ... colums. The colums headers are :  ......

Data Cleaning: No data cleaning step was required because the data was already in the format required. 
Data Quality Checks: Rows are completed, all datatype are corrected. 
The color colums has some missing values which was replaced wtih text "Not Specified". A new column "Color" was created.
```sql
Color = 
IF(
    OR(
        ISBLANK('myntra_products_catalog'[PrimaryColor]),              -- Check for blank values
        'myntra_products_catalog'[PrimaryColor] = ""                   -- Check for empty strings
    ),
    "Not Specified",
    'myntra_products_catalog'[PrimaryColor]
)
```

A new column called Gender Groud was created to group the gender into three groups 
```sql
GenderGroup = 
SWITCH(
    TRUE(),
    'myntra_products_catalog'[Gender] = "Men" || 'myntra_products_catalog'[Gender] = "Boy", "Male",
    'myntra_products_catalog'[Gender] = "Women" || 'myntra_products_catalog'[Gender] = "Girls", "Female",
    'myntra_products_catalog'[Gender] = "Unisex" || 'myntra_products_catalog'[Gender] = "Unisex Kids", "Unisex",
    BLANK()
)
```


Project Goal: The goal of this dashboard is for the Myntra Product Catalog manger to keep an eye on thier current current catalog and be able to easilty answer questions such as 1. How many product does Myntra currently have? 2. Which gender does their current catalog cater to. 3. How many brands do they have stocked. Which brand has trhe mopst expensive product? Which brand has the highest total sales value? 

This will help decide if they want to add more brands or reduce?


# Analysis

#### This dashboard is divided into 4 pages
* Pages 1: Shows the summary page which displays high level overview of the rest of the dashboar. From this page, the user can drill to other analysis and detail pages.


  ![SummaryPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Summary%20Page.PNG)

The number of distinct product was caluated by counting each distinct value in the Product ID column.
  ```sql
    DistinctProduct = DISTINCTCOUNT(myntra_products_catalog[ProductID])
  ```

To find the brand with the total highest sales value and the value, a few measure were created.
**MostExpensiveBrand**
```sql
MostExpensiveBrand = 
VAR MEB_Table = 
TOPN(
    1, 
    SUMMARIZE(
        myntra_products_catalog, 
        myntra_products_catalog[ProductBrand], 
        "TotalPrice", 
        SUM(myntra_products_catalog[Price (INR)])
    ), 
    [TotalPriceByBrand], 
    DESC
)
RETURN
    MAXX(MEB_Table, 'myntra_products_catalog'[ProductBrand])
```
**TotalPriceByBrand**
``` sql
TotalPriceByBrand = 
SUMX(
    VALUES(myntra_products_catalog[ProductBrand]),
    CALCULATE(SUM(myntra_products_catalog[Price (INR)]))
)
```

To find the brand with the most expensive proct, a few measures were created

**MaxPricePerBrand**
```sql
MaxPricePerBrand = 
CALCULATE(
    MAX('myntra_products_catalog'[Price (INR)]),
    ALLEXCEPT('myntra_products_catalog', 'myntra_products_catalog'[ProductBrand])
)

**MostExpensiveItemBrandName**
```sql
MostExpensiveItemBrandName = 
VAR MostExpensiveBrandTable = 
    TOPN(
        1,
        SUMMARIZE(
            'myntra_products_catalog',
            'myntra_products_catalog'[ProductBrand],
            "MaxPrice", [MaxPricePerBrand]
        ),
        [MaxPricePerBrand], DESC
    )
RETURN
    MAXX(MostExpensiveBrandTable, 'myntra_products_catalog'[ProductBrand])

```
To create the Most Frequent Brtand By erach Gender Table, these measures where created
**MostFrequentBrandByGender**
```sql
MostFrequentBrandByGender = 
CALCULATE(
    FIRSTNONBLANK('myntra_products_catalog'[ProductBrand], 1),
    FILTER(
        'myntra_products_catalog',
        [TotalProductCount] = [MaxProductCountByGender]
    )
)
```

**ProductCountForMostFrequentBrand**
```sql
ProductCountForMostFrequentBrand = 
CALCULATE(
    [TotalProductCount],
    FILTER(
        'myntra_products_catalog',
        'myntra_products_catalog'[ProductBrand] = [MostFrequentBrandByGender]
    )
)

```

To plot Top 10 brands by prodcut freqency, a new table was created 

```sql

BrandCounts = 
SUMMARIZE(
    'myntra_products_catalog',
    'myntra_products_catalog'[ProductBrand],
    "BrandCount", COUNT('myntra_products_catalog'[ProductBrand])
)
```

* Pages 2: This is first analysis page. It focuses on the Product Analysis. 
![ProductAnalysisPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Color%20Analysis%20Page.PNG)


* Page 3: This is the second analysis page: It focuses on the Color Analysis.
![ColorAnalysisPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Product%20Analysis%20Page.PNG)

*Page 4: This is the detail page. The user drill through to this page when they need fne grain leve details
![DetailPage]




*FYI: the anlysis on in the project was restricted by what was available in the data. No attempt was made to source for additional external data. 
