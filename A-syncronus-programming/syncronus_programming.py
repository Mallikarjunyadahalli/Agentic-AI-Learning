
import time

def fetch_weather():
    print("Fetching weather data...")
    time.sleep(4)  # Simulate a delay in fetching data
    print("Weather data fetched successfully!")

def fetch_news():
    print("Fetching news data...")
    time.sleep(2)  # Simulate a delay in fetching data
    print("News data fetched successfully!")


def main():
    start_time = time.time()
    print("Starting data fetch...")
    fetch_weather()
    fetch_news()
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("Data fetch completed!")


main()