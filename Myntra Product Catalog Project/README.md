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
   
This will help the  merchandizing manager or the floor maanger to make decisions about inventory management. . 


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
**Pages 1**: Shows the overview page which displays high level overview of the rest of the dashboard. From this page, the user can drill to other analysis and detail pages.
  ![OverviewPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Overview%20Page.PNG)

**Pages 2**: This is first analysis page. It focuses on the Product Analysis. 
![ProductAnalysisPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Product%20Analysis%20Page.PNG

**Page 3**: This is the second analysis page: It focuses on the Color Analysis.
![ColorAnalysisPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Color%20Analysis%20Page.PNG)

**Page 4**: This is the detail page. The user drills through to this page when they need fine grain level details
![DetailPage](https://github.com/Kosisochi/DataAnalysisPortfolio/blob/main/Myntra%20Product%20Catalog%20Project/images/Details%20Page.PNG)


# Measures and Columns
To add more data to enable a robust anlysis, i created a few new columns. 

**Product Addition Date**
This column specifies the date a prodcut was added to the catalog. 
```sql
Product Addition Date = 
VAR MinDate = DATE(2020, 1, 1)  -- Set your minimum date
VAR MaxDate = DATE(2025, 12, 31) -- Set your maximum date
VAR RandomDays = INT(RAND() * (MaxDate - MinDate) + 1)
RETURN
    MinDate + RandomDays
```

**Units Sold**
This column specifies the number of units for each product sold. 
```sql
Units Sold = 
VAR MinVal = 200  -- Set your minimum 
VAR MaxVal = 750 -- Set your maximum 
VAR RandomVal = INT(RAND() * (MaxVal - MinVal) + 1)
RETURN
    MinVal + RandomVal
```

**Units Returned**
This column specifies the numbver of units returned for each product. 
```sql 
Units Returned = 
VAR RandomNumber = 0.1 + (RAND() * (0.4 - 0.1))
RETURN
    myntra_products_catalog[Units Sold] * RandomNumber
```

**Return Rate**
```sql 
Return Rate = myntra_products_catalog[Units Returned] / myntra_products_catalog[Units Sold] 
```

**Revenue**
```sql
Revenue = myntra_products_catalog[Price (INR)] * myntra_products_catalog[Units Sold]
```

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

**Highest Revenue Brand**
```sql
MostRevenueBrand = 
VAR MRB_Table = 
TOPN(
    1, 
    SUMMARIZE(
        myntra_products_catalog, 
        myntra_products_catalog[ProductBrand], 
        "TotalRevenue", 
        SUM(myntra_products_catalog[Revenue])
    ), 
    [TotalRevenueByBrand], 
    DESC
)
RETURN
    MAXX(MRB_Table, 'myntra_products_catalog'[ProductBrand])
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

# Findings and Recommendations
1. Indian Terrain is most expensive brand and also generates the highest revenue for Myntra.
2. Garmin is the brand that has the most exspensive item but it doesn't generate the highest revenue most likely because the product does not sell very frequently.
3. Myntra has more products catering to women than any other gender or age group.
4. Products with Blue as its main color were more dominant in the product catalog reagrdless of the gender and age group.
5. Also, products with blue as its main color has the highest return rate. This could be because most of the producst are blue rather than that customers do not like purchasing blue clothing.
6. The number of products in Myntras catalog maintained a steady growth but saw a plummet in 2022 and 2025.
7. Out of over 12k product, only 894 did not have their dominant color specified. These could be floral or heavily patterned clothing without any outstanding main color. 

**FYI**
Columns like Revenue, Product Addition Date, Units Sold and Units Returned are randomized and will have a new value each time the dahboard refreshed. This should account for an discrepancies between the dashboard and any analysis in this readme page. 

