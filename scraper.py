import requests
from bs4 import BeautifulSoup

def scrape_website(url, output_file_path='output_file.txt'):
    # Send an HTTP request to the website
    response = requests.get(url)
    print("Console hiii")
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find and extract all text content
        text_content = soup.get_text(separator='\n')  # Use newline as the separator

        # Remove extra newline characters
        text_content = '\n'.join(line.strip() for line in text_content.splitlines() if line.strip())

        # Write the scraped text to the output file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(text_content)
        
        # Return the path of the output file
        return output_file_path
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
        return None
