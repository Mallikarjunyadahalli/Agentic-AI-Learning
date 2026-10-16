import asyncio
import time

async def fetch_weather():
    print("Fetching weather data...")
    await asyncio.sleep(4)  # Simulate a delay in fetching data
    print("Weather data fetched successfully!")

async def fetch_news():
    print("Fetching news data...")
    await asyncio.sleep(2)  # Simulate a delay in fetching data
    print("News data fetched successfully!")


async def main():
    start_time = asyncio.time()
    print("Starting data fetch...")
    await asyncio.gather(fetch_weather(), fetch_news())
    end_time = asyncio.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("Data fetch completed!")


    await main()