import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def get_phone_numbers(url):
    try:
        # Send a GET request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        phone_entries = []
        
        # Find all country-box divs
        country_boxes = soup.find_all('div', class_='country-box')
        
        for box in country_boxes:
            # Extract phone number
            phone_number = box.find('h4', class_='card-title').text.strip()
            
            # Extract time posted
            time_text = box.find('div', class_='add_time-top').text.strip()
            
            # Convert relative time to approximate datetime
            posted_time = parse_relative_time(time_text)
            
            phone_entries.append({
                'phone_number': phone_number,
                'time_posted': time_text,
                'estimated_datetime': posted_time.strftime('%Y-%m-%d %H:%M:%S') if posted_time else 'Unknown'
            })
        
        return phone_entries
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return []

def parse_relative_time(time_str):
    now = datetime.now()
    
    time_str = time_str.lower()
    if 'ago' in time_str:
        if 'minute' in time_str:
            minutes = int(time_str.split()[0])
            return now - timedelta(minutes=minutes)
        elif 'hour' in time_str:
            hours = int(time_str.split()[0])
            return now - timedelta(hours=hours)
        elif 'day' in time_str:
            days = int(time_str.split()[0])
            return now - timedelta(days=days)
    return None

# URL to scrape
url = 'https://temp-number.com/countries/United-States'

# Get phone numbers and their posted times
phone_numbers = get_phone_numbers(url)

# Print the results
if phone_numbers:
    print("Phone Numbers and Their Posting Times:")
    print("-" * 60)
    for entry in phone_numbers:
        print(f"Phone: {entry['phone_number']}")
        print(f"Posted: {entry['time_posted']}")
        print(f"Estimated DateTime: {entry['estimated_datetime']}")
        print("-" * 60)
else:
    print("No phone numbers found or there was an error fetching the data.")
