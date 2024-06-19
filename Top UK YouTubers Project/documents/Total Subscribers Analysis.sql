/*

1. Define Variables
2. Create a CTE that rounds the average views per video
3. Select the colums that are required for the analysis
4. Filter the results by the YouTube channels with the highest subscriber bases
5. Order by net_profit (from highest to lowest)

*/


--1. Define Variables
DECLARE @conversionRate FLOAT = 0.02;   -- Conversion Rate = 2%
DECLARE @productCost MONEY = 5.0;       -- Prodcut Cost = $5
DECLARE @campaignCost MONEY = 50000.0;  -- Campaign Cost = $50,000


--2. Create CTE
WITH ChannelData AS (
	SELECT 
		channel_name, 
		total_views, 
		total_videos, 
		ROUND((CAST(total_views AS  FLOAT)/total_videos), -4) AS rounded_avg_views_per_video
	FROM
		dbo.VIEW_TopUkYoutubers2024
)

--SELECT * FROM ChannelData




--3. Select the colums that are required for the analysis
SELECT 
	channel_name, 
	rounded_avg_views_per_video,
	(rounded_avg_views_per_video * @conversionRate) AS Potential_unit_sold_per_video,
	(rounded_avg_views_per_video * @conversionRate * @productCost) AS potentai_revenue_per_video,
	(rounded_avg_views_per_video * @conversionRate * @productCost) - @campaignCost AS net_profit
FROM ChannelData
--4. Filter the results by the YouTube channels with the highest subscriber bases
WHERE channel_name IN ('NoCopyrightSounds', 'DanTDM', 'Dan Rhodes')
--5. Order by net_profit (from highest to lowest)
ORDER BY net_profit DESC