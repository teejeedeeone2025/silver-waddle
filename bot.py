from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from datetime import datetime
import urllib.error
import vonage

# Vonage (Nexmo) SMS Configuration
VONAGE_API_KEY = "143a7dc2"
VONAGE_API_SECRET = "IB7ARMn3j5TKnBVg"
SMS_FROM = "Teejeedee"  # Your sender name
SMS_TO = "2349060558418"  # Your recipient number (international format)

def send_sms_notification(new_numbers):
    try:
        client = vonage.Client(key=VONAGE_API_KEY, secret=VONAGE_API_SECRET)
        sms = vonage.Sms(client)
        
        # Format the message
        message = f"New temp numbers found ({len(new_numbers)}):\n"
        for number in new_numbers[:10]:  # Limit to first 3 numbers due to SMS length limits
            message += f"{number['phone_number']} ({number['time_posted']})\n"
        
        if len(new_numbers) > 10:
            message += f"+{len(new_numbers)-3} more..."
        
        # Send SMS
        response = sms.send_message({
            "from": SMS_FROM,
            "to": SMS_TO,
            "text": message
        })
        
        if response["messages"][0]["status"] == "0":
            print("SMS sent successfully")
            return True
        else:
            print(f"SMS failed: {response['messages'][0]['error-text']}")
            return False
            
    except Exception as e:
        print(f"Error sending SMS: {e}")
        return False

def get_phone_numbers(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        }
        
        req = Request(url, headers=headers)
        response = urlopen(req, timeout=10)
        html_content = response.read().decode('utf-8')
        
        soup = BeautifulSoup(html_content, 'html.parser')
        phone_entries = []
        country_boxes = soup.find_all('div', class_='country-box')
        
        for box in country_boxes:
            try:
                phone_number = box.find('h4', class_='card-title').text.strip()
                time_text = box.find('div', class_='add_time-top').text.strip()
                
                phone_entries.append({
                    'phone_number': phone_number,
                    'time_posted': time_text
                })
            except AttributeError:
                continue
        
        return phone_entries
    
    except urllib.error.URLError as e:
        print(f"Error fetching the URL: {e.reason}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def check_for_new_numbers():
    url = 'https://temp-number.com/countries/United-States'
    print(f"Checking for new numbers at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    phone_numbers = get_phone_numbers(url)
    new_numbers = []
    
    for number in phone_numbers:
        time_text = number['time_posted'].lower()
        if 'minute' in time_text:
            minutes_ago = int(time_text.split()[0])
            if minutes_ago <= 2:
                new_numbers.append(number)
        elif 'second' in time_text:
            new_numbers.append(number)
    
    if new_numbers:
        print(f"Found {len(new_numbers)} new number(s) posted within last 2 minutes")
        return send_sms_notification(new_numbers)
    else:
        print("No new numbers found in the last 2 minutes")
        return False

if __name__ == "__main__":
    check_for_new_numbers()
