

Select location, date, total_cases, new_cases, total_deaths, population
From PortfolioProject.dbo.CovidDeaths
order by 1, CONVERT(DATE, date, 103) --CAST(date AS DATE)

--Select *
--From PortfolioProject.dbo.CovidVaccinations
--order by 3, 4

Select *
From PortfolioProject.dbo.CovidDeaths
where location like '%asia%' and continent is not null
order by 2, 3

 -- Shows likelihood of dying if you contracted covid in your country
Select location, date, total_cases, total_deaths, (CONVERT(FLOAT, total_deaths) /NULLIF(CONVERT(float,total_cases),0)) * 100 as DeathPercentage
From PortfolioProject.dbo.CovidDeaths
where location like '%Canada%'
order by 1, CONVERT(DATE, date, 103)


-- Looking at the total cases vs the population
-- Shows what percentage of the population has gotten Covid
Select location, date, total_cases, population, (NULLIF(CONVERT(float,total_cases),0) / CONVERT(FLOAT, population)) * 100 as InfectedPopulation
From PortfolioProject.dbo.CovidDeaths
where location like '%Canada%'
order by 1, CONVERT(DATE, date, 103)


--Looking at countries with highest infection rate compared with population
Select location, population, MAX(total_cases) as HighestInfectionCount, MAX((NULLIF(CONVERT(float,total_cases),0) / CONVERT(FLOAT, population)) * 100) as InfectedPopulation
From PortfolioProject.dbo.CovidDeaths
group by location, population
order by  InfectedPopulation desc

--Showing countries with highest death count per populatiopn 
Select location , MAX(cast(total_deaths as int)) as TotalDeathCount
From PortfolioProject.dbo.CovidDeaths
where continent <> '' --is not null
group by location
order by TotalDeathCount desc

-- LETS BREAK IT DOWN BY CONTINENT


--showing continents witht he highest death count per population
Select continent , MAX(cast(total_deaths as int)) as TotalDeathCount
From PortfolioProject.dbo.CovidDeaths
where continent <> '' --is not null
group by continent
order by TotalDeathCount desc

--Looking at continents with highest infection rate compared with population
Select continent,location, population, MAX(total_cases) as HighestInfectionCount, MAX((NULLIF(CONVERT(float,total_cases),0) / CONVERT(FLOAT, population)) * 100) as InfectedPopulation
From PortfolioProject.dbo.CovidDeaths
where continent <> '' --is not null
group by continent,location, population
order by  InfectedPopulation desc


--GLOBAL NUMBERS
Select location, date, total_cases, total_deaths, (CONVERT(FLOAT, total_deaths) /NULLIF(CONVERT(float,total_cases),0)) * 100 as DeathPercentage
From PortfolioProject.dbo.CovidDeaths
where location like '%Canada%'
order by 1, CONVERT(DATE, date, 103)


Select date, 
SUM(cast(new_cases as int)) as TotalCases, 
SUM(cast(new_deaths as int)) as TotalDeaths, 
case when SUM(cast(new_cases as int)) <> 0 then SUM(cast(new_deaths as int))  * 100.0 /SUM(cast(new_cases as int)) else Null end as DeathPercentage
From PortfolioProject.dbo.CovidDeaths
where continent <> '' --is not null
group by date
order by 1, CONVERT(DATE, date, 103)


Select
SUM(cast(new_cases as int)) as TotalCases, 
SUM(cast(new_deaths as int)) as TotalDeaths, 
case when SUM(cast(new_cases as int)) <> 0 then SUM(cast(new_deaths as int))  * 100.0 /SUM(cast(new_cases as int)) else Null end as DeathPercentage
From PortfolioProject.dbo.CovidDeaths
where continent <> '' --is not null
order by 1


-- Looking at Total Populations vs Vaccinations
Select CD.continent, 
CD.location, 
CONVERT(DATE, CD.date, 103), 
CD.population, 
CV.new_vaccinations, 
SUM(cast(CV.new_vaccinations as bigint)) OVER (Partition by CD.location order by CD.location, CONVERT(DATE, CD.date, 103)) as RollingNewVac

From PortfolioProject.dbo.CovidDeaths CD 
Join PortfolioProject.dbo.CovidVaccinations CV
on CD.location = CV.location
and  CONVERT(DATE, CD.date, 103) = CONVERT(DATE, CV.date, 103) 
where CD.continent <> '' --is not null
order by 1,2,3


-- Using CTE to find the number of people vacinnated out of the total population  ie RollingNEwVac/Population
With PopvsVac (continent, location, date, population, new_vaccinations, RollingNewVac) 
as 
(
Select CD.continent, 
CD.location, 
CONVERT(DATE, CD.date, 103), 
CD.population, 
CV.new_vaccinations, 
SUM(cast(CV.new_vaccinations as bigint)) OVER (Partition by CD.location order by CD.location, CONVERT(DATE, CD.date, 103)) as RollingNewVac

From PortfolioProject.dbo.CovidDeaths CD 
Join PortfolioProject.dbo.CovidVaccinations CV
on CD.location = CV.location
and  CONVERT(DATE, CD.date, 103) = CONVERT(DATE, CV.date, 103) 
where CD.continent <> '' --is not null
--order by 1,2,3
)
Select * , (RollingNewVac/Population) * 100 as PercentVaccinated
from PopvsVac


-- Using temptable to find the number of people vacinnated out of the total population  ie RollingNEwVac/Population
drop table if exists #PercentPopulationVaccinated
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255), 
Location nvarchar(255),
Date datetime, 
Population bigint, 
New_Vaccinations bigint,
RollingNewVac numeric
)

Insert into #PercentPopulationVaccinated

Select CD.continent, 
CD.location, 
CONVERT(DATE, CD.date, 103), 
TRY_CAST(REPLACE(CD.population, ',', '') AS bigint),
--TRY_CAST(CD.population as numeric), 
TRY_CAST(REPLACE(CV.new_vaccinations, ',', '') AS bigint),
SUM(TRY_CAST(CV.new_vaccinations as bigint)) OVER (Partition by CD.location order by CD.location, CONVERT(DATE, CD.date, 103)) as RollingNewVac

From PortfolioProject.dbo.CovidDeaths CD 
Join PortfolioProject.dbo.CovidVaccinations CV
on CD.location = CV.location
and  CONVERT(DATE, CD.date, 103) = CONVERT(DATE, CV.date, 103) 
where CD.continent <> '' --is not null
--order by 1,2,3

Select * , (RollingNewVac/Population) * 100 as PercentVaccinated
from #PercentPopulationVaccinated


-- Create View to Stor dat for later visualizations
Create View PercentPopulationVaccinated as 
Select CD.continent, 
CD.location, 
CONVERT(DATE, CD.date, 103) AS Date, 
TRY_CAST(REPLACE(CD.population, ',', '') AS bigint) AS Population,
TRY_CAST(REPLACE(CV.new_vaccinations, ',', '') AS bigint) AS NewVaccinations,
SUM(TRY_CAST(CV.new_vaccinations as bigint)) OVER (Partition by CD.location order by CD.location, CONVERT(DATE, CD.date, 103)) as RollingNewVac

From PortfolioProject.dbo.CovidDeaths CD 
Join PortfolioProject.dbo.CovidVaccinations CV
on CD.location = CV.location
and  CONVERT(DATE, CD.date, 103) = CONVERT(DATE, CV.date, 103) 
where CD.continent <> '' --is not null


Select * 
From PercentPopulationVaccinated