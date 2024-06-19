/*
DATA CLEANING STEPS
1. Remove columns not required
2. Extract Channle names from the first colum ie. selecte text before "
3. Rename Column names to English

*/


CREATE VIEW VIEW_TopUkYoutubers2024 AS

SELECT 
	CAST(SUBSTRING(NOMBRE, 1 , CHARINDEX('@', NOMBRE) -1)  AS VARCHAR(100)) AS channel_name,
	total_subscribers,
	total_videos, 
	total_views
FROM dbo.TopUkYoutubers2024


/*
DATA QUALITY CHECKS
1. Row Count Test: 100 rows ie. 100 100 youtube channel
2. Column Count Test: 4 columns
3. Data Type Test: "channel_name" column must be string while the other3 must be a numerical (integer) data type
4. Duplicity Test: Each record must be unique
*/


--ROW COUNT CHECK - passed
SELECT Count(*) as num_of_rows
FROM dbo.TopUkYoutubers2024

--COLUMN COUNT CHECK - passed
SELECT Count(*) AS colum_count
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'VIEW_TopUkYoutubers2024'

-- DATA TYPE CHECK - passed
SELECT column_name, data_type
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'VIEW_TopUkYoutubers2024'

-- DUPLICITY CHECK -- passed
SELECT
	channel_name, 
	Count(*) as duplicate_count
 FROM VIEW_TopUkYoutubers2024
 GROUP BY channel_name
 HAVING Count(*) > 1

