/*
Cleaning Data in SQL queries
*/

Select *
from PortfolioProject.dbo.HousingData

/* STANDARDIZE DATA FORMAT. it was in datetime format, we dont need the time  */

--first, we select the column and convert it to see if its done correctly
Select SaleDate, CONVERT(Date, SaleDate)
from PortfolioProject.dbo.HousingData

--then we upate the Date column in the table with the conversion
Update HousingData 
SET SaleDate = CONVERT(Date, SaleDate)

-- or 
ALTER TABLE HousingData 
ALTER COLUMN SaleDate DATE 

--Alternatively, we can create a new colum with the conversion 
ALTER TABLE HousingData
Add SaleDdateConveretd Date;

Update HousingData
SET SaleDateConverted = CONVERT (Date, SaleDate)


/* POPULATE PROPERTY ADDRESS */

Select *
from PortfolioProject.dbo.HousingData
where PropertyAddress is null
-- we found out that ParcelId column has a one to one relationship with the PropertyAddress coulum, so we will use that to popluate it.

-- here we write the code to check
Select A.Parcel.ID, A.PropertyAddress,  B.ParcelID , B.PropertyAddress, ISNULL(A.PropertyAddress, B.PropertyAddress) --this checks if A.propoertyadress is null, if yes, replace it with B.PropertyAddress)
from PortfolioProject.dbo.HousingData  A
JOIN PortfolioProject.dbo.HousingData  B
	on A.Parcel.ID = B.ParcelID
	AND A.UniqueID <> B.UniqueID
where A.PropertyAddress is NULL

--then we write the code to do actual updating
Update A
SET PropertyAddress =  ISNULL(A.PropertyAddress, B.PropertyAddress) 
from PortfolioProject.dbo.HousingData  A
JOIN PortfolioProject.dbo.HousingData  B
	on A.Parcel.ID = B.ParcelID
	AND A.UniqueID <> B.UniqueID
where A.PropertyAddress is NULL



/* BREAKING OUT ADDRESS INTO INDIVIDUAL COLUMS( Address, City, State) */

Select PropertyAddress
from PortfolioProject.dbo.HousingData

Select
Substring(PropertyAddress, 1, CHARINDEX(',' , PropertyAddress) - 1) as PropertySplitAddress
--propertyadrress, position 1, look for  a comma
--CHARINDEX return the index for the specified dstring, so in this case, it returns the index of the comma
--Example, if CHARINDEX returns 19, then Substring(Property Adress, 1 ,19) will return the substring from index 1 to 19.
-- CHARINDEX(',' , PropertyAddress) - 1) get the index - 1 
, Substring(PropertyAddress, CHARINDEX(',' , PropertyAddress) + 1, LEN(PropertyAddress))  as PropertySplitCity
from PortfolioProject.dbo.HousingData


--next we create two new colums to add the seperated address in  
ALTER TABLE HousingData
Add PropertySplitAddress Nvarchar(255)

UPDATE HousingData
SET PropertySplitAddress = Substring(PropertyAddress, 1, CHARINDEX(',' , PropertyAddress) - 1)


ALTER TABLE HousingData
Add PropertySplitCity Nvarchar(255)

UPDATE HousingData
SET PropertySplitCity = Substring(PropertyAddress, CHARINDEX(',' , PropertyAddress) + 1, LEN(PropertyAddress))


/* Owner Address -- using PARSENAME rather than SUBSTRING */

Select OwnerAddress
From HousingData

--PARSENAME only works with periods. so we can replace the commas with period first
-- PARSENAME will return the text with period as a delimiter
Select 
PARSENAME(REPLACE(OwnerAddress,',', '.') , 1) -- 1 means the last item
,PARSENAME(REPLACE(OwnerAddress,',', '.') , 2) --  2 means the 2nd to last
,PARSENAME(REPLACE(OwnerAddress,',', '.') , 3) -- 3 means the 3rd to last
From HousingData


ALTER TABLE HousingData
Add OwnerSplitAddress Nvarchar(255)

UPDATE HousingData
SET OwnerSplitAddress = PARSENAME(REPLACE(OwnerAddress,',', '.') , 3)

ALTER TABLE HousingData
Add OwnerSplitCity Nvarchar(255)

UPDATE HousingData
SET OwnerSplitCity = PARSENAME(REPLACE(OwnerAddress,',', '.') , 2)


ALTER TABLE HousingData
Add OwnerSplitState Nvarchar(255)

UPDATE HousingData
SET OwnerSplitState = PARSENAME(REPLACE(OwnerAddress,',', '.') , 1)



/* CHANGE Y and N to Yes and No in "Sold as Vacant" field to keep it all uniform    */

Select SoldAsVacant,
CASE when SoldAsVacant = 'Y' THEN 'Yes',
CASE WHEN SoldAsVacant = 'N' THEN 'No', 
ELSE SoldAsVacant
END
From  HousingData


UPDATE HousingData
SET SoldAsVacant = 
CASE WHEN SoldAsVacant = 'Y' THEN 'Yes',
CASE WHEN SoldAsVacant = 'N' THEN 'No', 
ELSE SoldAsVacant
END



/* Remove Duplicaties */

WITH RowNumCTE AS (
----- this code below will create a column called row_num where unique rows will have value 1 and duplicate rows witrh have 1, 2 ,3 dependinf on how many they are. 
Select * , 
ROW_NUMBER() OVER (
PARTITION BY ParcelID, 
PropetyAddress,
SalePRice,
SaleDate, 
LegalReference
ORDER BY UniqueID ) row_num

from HousingData
--Order by ParcelID
)

-- this shows rows that are duplicates 
Select * 
From RowNumCTE
where row_num > 1 
order by PropertyAddress

-- this then deletes the duplicates
Delete * 
From RowNumCTE
where row_num > 1 



/* Delete Unused Colums */
ALTER HousingData
DROP COLUM OwnerAddress, TaxDistrict, PropertyAddress