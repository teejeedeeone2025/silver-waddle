from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import chromedriver_autoinstaller
import time

def run_mtn_automation():
    # Install ChromeDriver
    chromedriver_autoinstaller.install()

    # Configure Chrome options
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920x1080')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

    # Initialize the driver
    driver = webdriver.Chrome(options=options)

    try:
        # Navigate to the MTN login page
        url = "https://auth.mtnonline.com/login?state=hKFo2SBCSjVFajhyM211MDdIX1VXdUdLSzlFeUJTb1o3LUVYU6FupWxvZ2luo3RpZNkgc0NDdzczeFNwbl8zTjN5bkpuY3pQVXM5Sk9rVHZ0TFajY2lk2SB0V05sSkJmcXY4QjVjOXJWcml5OUVhdkVaTjN6cjQ2NQ&client=tWNlJBfqv8B5c9rVriy9EavEZN3zr465&protocol=oauth2&redirect_uri=https%3A%2F%2Fshop.mtn.ng%2Fmtnng%2Feshop%2Fcallback%2F&scope=openid%20profile%20email&response_mode=query&response_type=code&nonce=252325dd4d7509175e0bbd50e6d7ee0d&code_challenge=Ke-Ktp4CWM3tX2R_jGmLvDXkltKCZQrNFqGYrSueOAI&code_challenge_method=S256&theme="
        driver.get(url)

        # Wait for the page to load
        time.sleep(3)

        # Get the body element to send keys to
        body = driver.find_element(By.TAG_NAME, 'body')

        # Press Tab 7 times to reach the desired button
        for _ in range(7):
            body.send_keys(Keys.TAB)
            time.sleep(0.3)

        # Get the currently focused button
        button = driver.switch_to.active_element

        # Click the button directly
        button.click()
        print("Clicked the button directly")
        time.sleep(0.3)

        # Press Tab to move to the input field
        body.send_keys(Keys.TAB)
        time.sleep(0.5)

        # Get the currently focused input field
        input_field = driver.switch_to.active_element

        # Type the phone number directly
        phone_number = "09060558418"
        input_field.send_keys(phone_number)
        time.sleep(1)

        # Press Enter to submit
        input_field.send_keys(Keys.ENTER)
        time.sleep(0.5)

        # Take a screenshot
        driver.save_screenshot('final_result.png')
        print("Screenshot saved as 'final_result.png'")

        # Keep browser open for observation
        time.sleep(5)

    finally:
        driver.quit()

if __name__ == "__main__":
    run_mtn_automation()
