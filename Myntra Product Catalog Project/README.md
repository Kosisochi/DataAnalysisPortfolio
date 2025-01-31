# Data Visualization Project: PowerBI


## Skills Required
- PowerBI
- DAX

#  Project Title: Myntra Product Catalog. 
Myntra is a major Indian fashion e-commerce company headquartered in Bengaluru, Karnataka, India. The company was founded in 2007 to sell personalized gift items. In May 2014, Myntra.com was acquired by Flipkart.
It sells clothing items from multiple curated brands. It services women, men, girls, boys and unisex for both adults and kids. 

# Project Goal:
The goal of this dashboard is for the Myntra Product Catalog Manager to monitor their current catalog and easily answer questions such as
1. How many products does Myntra currently have?
2. Which gender does thier current catalog cater to?
3. How many brands do they have stocked?
4. Which brand has the most expensive product?
5. Which brand has the highest total sales value?
   
This will help in making multiple decisions. 


# Data:
**Data Source**: The data was downloaded from Kaggle [Click Here To Download](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog?resource=download)

**Data Description**: It contains 12491 rows and 8 columns. The column headers are : Product ID (unique column), Price INR, Gender, Primary Color, ProductName, ProductBrand, NumImages, Description

**Data Cleaning**: No data cleaning steps were required because the data was already in a clean format. 

**Data Quality Checks**: Rows are complete, all datatypes are correct. 
The Primary Color column has some missing values which i replaced with text "Not Specified". A new column "Color" was created rather than replacing the Primary Color column.

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

A new column called Gender Group was created to group the gender into 3 major groups: Male, Female & Unisex

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


# Dashboard

[**PBIX PROJECT FILE**](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/Myntra%20Product%20Catalog.pbix)

### This dashboard is divided into 4 pages
**Pages 1**: Shows the summary page which displays high level overview of the rest of the dashboard. From this page, the user can drill to other analysis and detail pages.
  ![SummaryPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Summary%20Page.PNG)

**Pages 2**: This is first analysis page. It focuses on the Product Analysis. 
![ProductAnalysisPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Color%20Analysis%20Page.PNG)

**Page 3**: This is the second analysis page: It focuses on the Color Analysis.
![ColorAnalysisPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Product%20Analysis%20Page.PNG)

**Page 4**: This is the detail page. The user drills through to this page when they need fine grain level details
![DetailPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Detail%20Page.PNG)


# Measures and Columns

The number of distinct products was calculated by counting each distinct value in the Product ID column.

**DistinctProduct**
  ```sql
    DistinctProduct = DISTINCTCOUNT(myntra_products_catalog[ProductID])
  ```

To find the brand with the total highest sales value and the corresponding value, a few measures were created (MostExpensiveBrand and TotalPriceByBrand).

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

To find the brand with the most expensive product, a few measures were created (MaxPricePerBrand and MostExpensiveItemBrandName)

**MaxPricePerBrand**
```sql
MaxPricePerBrand = 
CALCULATE(
    MAX('myntra_products_catalog'[Price (INR)]),
    ALLEXCEPT('myntra_products_catalog', 'myntra_products_catalog'[ProductBrand])
)
```

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

To create the Most Frequent Brand By each Gender Table, these measures where created.

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

To plot Top 10 brands by product frequency, a new table was created 

**BrandCounts**

```sql
BrandCounts = 
SUMMARIZE(
    'myntra_products_catalog',
    'myntra_products_catalog'[ProductBrand],
    "BrandCount", COUNT('myntra_products_catalog'[ProductBrand])
)
```

# Findings


**FYI**: The analysis caaried out in the project was restricted by what was available in the data. No attempt was made to source for additional external data. 
