import os
from mistralai import Mistral
import discord
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
import re
import json

MISTRAL_MODEL = "mistral-large-latest"
SYSTEM_PROMPT = """You are an accessibility assistant helping users with vision impairments navigate websites. 
Your role is to:
1. Describe web pages clearly and concisely, using a natural language and without any bullet points.
2. Help users navigate through websites, click on links, and navigate through pages.
3. Describe couple of images in every page, and other images when specifically requested
4. Suggest possible actions users can take
5. Always ask users what they would like to do next

Keep responses clear, informative, and focused on helping users navigate effectively."""

class AccessibilityAgent:
    def __init__(self):
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        
        # Initialize Azure Computer Vision client
        vision_key = os.getenv("AZURE_VISION_KEY")
        vision_endpoint = os.getenv("AZURE_VISION_ENDPOINT")
        self.vision_client = ComputerVisionClient(
            vision_endpoint,
            CognitiveServicesCredentials(vision_key)
        )
        
        # Initialize Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        
        self.current_url = None
        self.current_page_content = None
        self.conversation_history = []
        self.status_message = None
        self.navigation_history = []
        self.current_section = None

    def normalize_text(self, text):
        """Normalize text for comparison"""
        return ' '.join(text.lower().split())

    def find_best_match(self, target, candidates, threshold=0.5):
        """Find the best matching text from a list of candidates"""
        target = self.normalize_text(target)
        best_match = None
        best_score = 0
        
        for candidate in candidates:
            candidate_text = self.normalize_text(candidate)
            # Check exact match first
            if target == candidate_text:
                return candidate, 1.0
            
            # Check if target words are subset of candidate words
            target_words = set(target.split())
            candidate_words = set(candidate_text.split())
            
            # Calculate word overlap ratio
            if target_words:
                overlap = len(target_words & candidate_words) / len(target_words)
                if overlap > best_score and overlap >= threshold:
                    best_score = overlap
                    best_match = candidate
        
        return best_match, best_score

    async def set_status(self, channel, message):
        """Update or create a status message"""
        if self.status_message:
            try:
                await self.status_message.delete()
            except:
                pass
        self.status_message = await channel.send(f"🔄 {message}")

    async def clear_status(self):
        """Clear the current status message"""
        if self.status_message:
            try:
                await self.status_message.delete()
            except:
                pass
            self.status_message = None

    async def describe_specific_images(self, channel, start_index=0, count=3):
        """Describe a specific set of images from the current page"""
        await self.set_status(channel, "🖼️ Analyzing images...")
        
        if not self.current_page_content or 'images' not in self.current_page_content:
            await self.clear_status()
            return "No images available on the current page."
        
        images = self.current_page_content['images']
        if not images:
            await self.clear_status()
            return "No images found on the current page."
        
        end_index = min(start_index + count, len(images))
        if start_index >= len(images):
            await self.clear_status()
            return "No more images available on this page."
        
        descriptions = []
        for i, img in enumerate(images[start_index:end_index], start=start_index + 1):
            if img['src'].startswith(('http://', 'https://')):
                await self.set_status(channel, f"🖼️ Analyzing image {i} of {end_index}...")
                desc = await self.describe_image(img['src'])
                if desc:
                    descriptions.append(f"Image {i}: {desc}")
        
        remaining = len(images) - end_index
        response = "\n".join(descriptions)
        if remaining > 0:
            response += f"\n\nThere are {remaining} more images available. You can ask me to describe more."
        
        await self.clear_status()
        return response

    async def describe_image(self, image_url):
        """Describe an image using Azure Computer Vision API"""
        try:
            description_result = self.vision_client.describe_image(image_url)
            if description_result.captions:
                return description_result.captions[0].text
            return "Unable to generate description for this image."
        except Exception as e:
            return f"Error describing image: {str(e)}"

    def extract_page_content(self):
        """Extract relevant content from the current page"""
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        
        content = {
            'title': self.driver.title,
            'headings': [],
            'links': [],
            'images': [],
            'main_text': {}
        }
        
        # Extract headings with their text and IDs
        for h in soup.find_all(['h1', 'h2', 'h3']):
            heading_text = h.text.strip()
            if heading_text:
                content['headings'].append({
                    'text': heading_text,
                    'level': int(h.name[1]),
                    'id': h.get('id', '')
                })
        
        # Extract links with better context and descriptions
        for link in soup.find_all('a'):
            link_text = link.text.strip()
            href = link.get('href', '')
            
            if link_text and href:
                # Get surrounding text for better context
                surrounding_text = ''
                if link.parent and link.parent.text:
                    full_text = link.parent.text.strip()
                    link_pos = full_text.find(link_text)
                    if link_pos > 0:
                        surrounding_text = full_text[:link_pos].strip()
                    if link_pos + len(link_text) < len(full_text):
                        surrounding_text += ' ' + full_text[link_pos + len(link_text):].strip()
                
                content['links'].append({
                    'text': link_text,
                    'description': surrounding_text,
                    'location': self.get_element_location_description(link)
                })
        
        # Extract images with their URLs and context
        for img in soup.find_all('img'):
            src = img.get('src', '')
            alt = img.get('alt', '')
            if src:
                content['images'].append({
                    'src': src if src.startswith(('http://', 'https://')) else f"{self.current_url.rstrip('/')}/{src.lstrip('/')}",
                    'alt': alt,
                    'context': img.parent.name if img.parent else 'Unknown'
                })
        
        # Extract main text content with headers
        current_section = "Main Content"
        content['main_text'][current_section] = []
        
        for elem in soup.find_all(['h1', 'h2', 'h3', 'p', 'div']):
            if elem.name in ['h1', 'h2', 'h3']:
                current_section = elem.text.strip()
                if current_section not in content['main_text']:
                    content['main_text'][current_section] = []
            elif elem.name in ['p', 'div']:
                text = ' '.join(elem.stripped_strings)
                if text and not any(text in existing for existing in content['main_text'][current_section]):
                    content['main_text'][current_section].append(text)
        
        return content

    def get_element_location_description(self, element):
        """Generate a descriptive location for an element"""
        location = []
        
        # Check if element is in header/footer/sidebar
        parent = element.parent
        while parent:
            if parent.name in ['header', 'nav', 'footer', 'aside']:
                location.append(f"in the {parent.name}")
                break
            parent = parent.parent
        
        # Check if element is in a list
        if element.find_parent(['ul', 'ol']):
            items = element.find_parent(['ul', 'ol']).find_all(['li'])
            for i, item in enumerate(items, 1):
                if element in item.descendants:
                    location.append(f"item {i} in a list")
                    break
        
        return ' '.join(location) if location else 'in main content'

    async def find_and_click_element(self, target_text):
        """Find and click an element using various strategies"""
        try:
            # Try direct link text first
            try:
                element = WebDriverWait(self.driver, 2).until(
                    EC.element_to_be_clickable((By.LINK_TEXT, target_text))
                )
                return element
            except:
                pass

            # Try partial link text
            try:
                element = WebDriverWait(self.driver, 2).until(
                    EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, target_text))
                )
                return element
            except:
                pass

            # Try finding by text content with case-insensitive comparison
            try:
                xpath = f"//*[translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='{target_text.lower()}' or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{target_text.lower()}')]"
                element = WebDriverWait(self.driver, 2).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                return element
            except:
                pass

            # Try finding by aria-label or title
            try:
                xpath = f"//*[@aria-label='{target_text}' or @title='{target_text}' or @alt='{target_text}']"
                element = WebDriverWait(self.driver, 2).until(
                    EC.element_to_be_clickable((By.XPATH, xpath))
                )
                return element
            except:
                pass

            return None
        except Exception as e:
            return None

    async def navigate_to_section(self, target_section):
        """Navigate to a specific section and extract its content"""
        try:
            # First try to find the section by exact heading match
            section_xpath = f"//h1[contains(text(), '{target_section}')] | //h2[contains(text(), '{target_section}')] | //h3[contains(text(), '{target_section}')]"
            try:
                section = WebDriverWait(self.driver, 2).until(
                    EC.presence_of_element_located((By.XPATH, section_xpath))
                )
                section.location_once_scrolled_into_view
                return self.extract_section_content(section)
            except:
                pass

            # If not found, try clicking a link to the section
            element = await self.find_and_click_element(target_section)
            if element:
                element.click()
                self.current_page_content = self.extract_page_content()
                return True

            return False
        except Exception as e:
            return False

    def extract_section_content(self, section_element):
        """Extract content from a specific section"""
        try:
            # Get the section's content
            section_content = {
                'title': section_element.text,
                'content': [],
                'links': [],
                'images': []
            }
            
            # Find all content elements after this heading until the next heading
            current = section_element
            while current := current.find_element_by_xpath('following-sibling::*[1]'):
                if current.tag_name in ['h1', 'h2', 'h3']:
                    break
                
                if current.tag_name in ['p', 'div']:
                    text = current.text.strip()
                    if text:
                        section_content['content'].append(text)
                
                # Extract links within the section
                links = current.find_elements_by_tag_name('a')
                for link in links:
                    link_text = link.text.strip()
                    if link_text:
                        section_content['links'].append({
                            'text': link_text,
                            'href': link.get_attribute('href')
                        })
                
                # Extract images within the section
                images = current.find_elements_by_tag_name('img')
                for img in images:
                    src = img.get_attribute('src')
                    alt = img.get_attribute('alt')
                    if src:
                        section_content['images'].append({
                            'src': src,
                            'alt': alt
                        })
            
            return section_content
        except Exception as e:
            return None

    async def navigate_to_url(self, channel, url):
        """Navigate to a URL and extract its content"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            await self.set_status(channel, f"📱 Loading {url}...")
            self.driver.get(url)
            self.current_url = url
            
            await self.set_status(channel, "📄 Analyzing page content...")
            self.current_page_content = self.extract_page_content()
            
            await self.clear_status()
            return self.current_page_content
        except Exception as e:
            await self.clear_status()
            return f"❌ Error navigating to URL: {str(e)}"

    def parse_user_command(self, user_input):
        """Parse and normalize user commands into actionable requests"""
        input_lower = self.normalize_text(user_input)
        
        # Special commands that don't need pattern matching
        special_commands = {
            'back': ['back', 'previous page', 'go back', 'return'],
            'describe_images': ['describe image', 'show image', 'what is in the image'],
            'summarize': ['summarize', 'summary', 'summarize this', 'summarize page', 'can you summarize']
        }
        
        # Check special commands first
        for cmd_type, phrases in special_commands.items():
            if any(phrase in input_lower for phrase in phrases):
                return {'type': cmd_type}
        
        # Command patterns
        section_patterns = [
            r'(?:can you )?(?:go|show|navigate|take) (?:me )?(?:to )?(?:the )?["\']?([^"\']+?)["\']? section',
            r'(?:show|display|read|view|open) (?:the )?["\']?([^"\']+?)["\']? section',
            r'["\']?([^"\']+?)["\']? section',
        ]
        
        navigation_patterns = [
            r'(?:can you )?(?:go|navigate|take) (?:me )?(?:to )?(?:the )?["\']?([^"\']+?)["\']?(?:\s|$)',
            r'(?:show|display|open) (?:the )?["\']?([^"\']+?)["\']?(?:\s|$)',
            r'(?:click|select|choose) (?:on )?(?:the )?["\']?([^"\']+?)["\']?(?:\s|$)',
        ]
        
        # Check for section requests
        for pattern in section_patterns:
            match = re.search(pattern, input_lower)
            if match:
                section_name = match.group(1).strip()
                return {
                    'type': 'section',
                    'target': section_name,
                    'original_text': section_name
                }
        
        # Check for navigation requests
        for pattern in navigation_patterns:
            match = re.search(pattern, input_lower)
            if match:
                target = match.group(1).strip()
                return {
                    'type': 'navigation',
                    'target': target,
                    'original_text': target
                }
        
        # If no specific command is recognized, return the normalized input
        return {
            'type': 'unknown',
            'original_text': user_input
        }

    def get_available_actions(self):
        """Get a list of currently available actions based on context"""
        actions = []
        
        if self.navigation_history:
            actions.append("Go back to the previous page")
        
        if self.current_page_content:
            if self.current_page_content.get('images'):
                actions.append("Get descriptions of the images on the page")
            
            if self.current_page_content.get('headings'):
                sections = [h['text'] if isinstance(h, dict) else h 
                          for h in self.current_page_content['headings']]
                if sections:
                    actions.append(f"Navigate to sections: {', '.join(sections[:3])}"
                                 + ("..." if len(sections) > 3 else ""))
            
            if self.current_page_content.get('links'):
                links = [link['text'] for link in self.current_page_content['links']]
                if links:
                    actions.append(f"Click on links like: {', '.join(links[:3])}"
                                 + ("..." if len(links) > 3 else ""))
        
        return actions

    def get_error_response(self, command_type, context=None):
        """Generate conversational error responses"""
        if command_type == 'back':
            if not self.navigation_history:
                return "This is actually the first page we've visited. Would you like to explore something here instead?"
        
        if command_type == 'section':
            if not self.current_page_content or not self.current_page_content.get('headings'):
                return "I don't see any specific sections on this page. What would you like to know about the content I can see?"
            sections = [h['text'] if isinstance(h, dict) else h 
                       for h in self.current_page_content['headings']]
            return f"I can see several sections here: {', '.join(sections)}. Which one interests you?"
        
        if command_type == 'navigation':
            if not self.current_page_content or not self.current_page_content.get('links'):
                return "I don't see any links we can click on this page. Is there something specific you're looking for?"
        
        # Default response
        return "I'm not quite sure what you'd like to do. We can explore the content here, look at images, or navigate to different pages. What interests you?"

    async def summarize_page(self, context):
        """Generate a brief, focused summary of the page content."""
        try:
            messages = [
                {"role": "system", "content": """You are a accessibility assistant having a conversation with someone who cannot see the website. 
                    Give them a quick overview and summary of the page in a natural daily language.
                    Keep it brief, focusing on what the page is about.
                    End with a natural suggestion about what they might want to explore first."""},
                {"role": "user", "content": f"Tell me about this webpage in a conversational way in summary:\n{json.dumps(context)}"}
            ]

            response = await self.mistral_client.chat.complete_async(
                model=MISTRAL_MODEL,
                messages=messages
            )

            if response and response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
                
            return "I'm having trouble generating a summary of this page. Would you like me to try again?"
        except Exception as e:
            return f"I encountered an error while summarizing the page: {str(e)}. How can I help you?"

    async def process_user_input(self, channel, user_input):
        """Process user input using Mistral AI for natural language understanding and action generation"""
        try:
            # Handle URL directly if present
            url_pattern = r'https?://\S+|(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}'
            url_match = re.search(url_pattern, user_input)
            if url_match:
                url = url_match.group(0)
                await self.set_status(channel, f"📱 Navigating to {url}...")
                page_content = await self.navigate_to_url(channel, url)
                if isinstance(page_content, dict):
                    return await self.get_compact_page_description(page_content)
                return f"❌ Couldn't access that website: {page_content}"

            if not self.current_page_content:
                return "👋 Share any website URL with me, and I'll help you explore it!"

            # Parse the user command first
            command = self.parse_user_command(user_input)
            
            # Handle summarize command directly
            if command['type'] == 'summarize':
                if not self.current_page_content:
                    return "No page is currently loaded. Please share a URL to explore."
                return await self.summarize_page(self.current_page_content)

            # For other inputs, use Mistral to understand intent
            messages = [
                {"role": "system", "content": """You are an accessibility assistant helping users navigate websites.
        Analyze the user's request and respond with a JSON object containing:
        {
            "action": one of ["describe_page", "describe_section", "describe_images", "click_link", "go_back", "answer_question"],
            "target": specific section/link/element to interact with (if applicable),
            "details": any additional context needed
        }"""},
                {"role": "user", "content": f"User request: {user_input}\nCurrent page content: {json.dumps(self.current_page_content)}"}
            ]

            intent_response = await self.mistral_client.chat.complete_async(
                model=MISTRAL_MODEL,
                messages=messages
            )
            
            try:
                intent = json.loads(intent_response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fallback to treating the input as a general question
                intent = {
                    "action": "answer_question",
                    "details": user_input
                }

            # Handle different actions based on intent
            if intent['action'] == 'navigate_to_url':
                url = intent['target']
                await self.set_status(channel, f"📱 Navigating to {url}...")
                page_content = await self.navigate_to_url(channel, url)
                if isinstance(page_content, dict):
                    return await self.get_compact_page_description(page_content)
                return f"❌ Couldn't access that website: {page_content}"

            elif intent['action'] == 'describe_page':
                if not self.current_page_content:
                    return "No page is currently loaded. Please share a URL to explore."
                return await self.get_compact_page_description(self.current_page_content)

            elif intent['action'] == 'describe_section':
                if not self.current_page_content:
                    return "No page is currently loaded. Please share a URL to explore."
                
                # Try to navigate to the section first
                await self.navigate_to_section(intent['target'])
                
                messages = [
                    {"role": "system", "content": """You are a accessibility assistant having a conversation with someone who cannot see the website. 
                    Describe what you see in a direct, natural way using daily language.

                    Keep descriptions conversational and focused. Don't list everything - let the user ask for more details about what interests them.
                    If you see interesting links or images, mention them naturally in your description."""},
                    {"role": "user", "content": f"Describe this section using daily language: {intent['target']}\nContent: {json.dumps(self.current_page_content)}"}
                ]
                
                response = await self.mistral_client.chat.complete_async(
                    model=MISTRAL_MODEL,
                    messages=messages
                )
                
                # Add a natural transition to encourage further exploration
                section_description = response.choices[0].message.content
                final_response = section_description + "\n\nWhat would you like to know more about?"
                return final_response

            elif intent['action'] == 'describe_images':
                return await self.describe_specific_images(channel)

            elif intent['action'] == 'click_link':
                if not self.current_page_content:
                    return "No page is currently loaded. Please share a URL to explore."
                
                element = await self.find_and_click_element(intent['target'])
                if element:
                    await self.set_status(channel, f"Navigating to '{intent['target']}'...")
                    element.click()
                    self.navigation_history.append(self.current_url)
                    self.current_page_content = self.extract_page_content()
                    self.current_url = self.driver.current_url
                    return await self.get_compact_page_description(self.current_page_content)
                return f"I couldn't find a link matching '{intent['target']}'. Could you try describing what you're looking for differently?"

            elif intent['action'] == 'go_back':
                if not self.navigation_history:
                    return "This is the first page we've visited. What would you like to explore here?"
                
                await self.set_status(channel, "Going back to previous page...")
                self.driver.back()
                self.current_url = self.navigation_history.pop()
                self.current_page_content = self.extract_page_content()
                return await self.get_compact_page_description(self.current_page_content)

            elif intent['action'] == 'answer_question':
                if not self.current_page_content:
                    return "No page is currently loaded. Please share a URL to explore."
                
                messages = [
                    {"role": "system", "content": """You are an accessibility assistant having a natural conversation with someone who cannot see the website. 
                    Answer questions in a direct, natural way using daily language.
                    Keep your responses conversational and focused on what the user wants to know.
                    If there's related information that might interest them, mention it briefly at the end."""},
                    {"role": "user", "content": f"Answer this question in a conversational way using daily language: {intent['details']}\nContent: {json.dumps(self.current_page_content)}"}
                ]
                
                response = await self.mistral_client.chat.complete_async(
                    model=MISTRAL_MODEL,
                    messages=messages
                )
                return response.choices[0].message.content

            return "I'm not sure what you'd like to do. You can ask me to describe the page, look at specific sections, describe images, or click on links."

        except Exception as e:
            await self.clear_status()
            return f"❌ Error: {str(e)}\nWhat would you like to know about the current page?"

    async def run(self, message: discord.Message):
        """Process user messages and generate responses"""
        user_input = message.content.strip()
        channel = message.channel
        
        # Check if it's a conversational prompt first
        if await self.is_conversational_prompt(user_input):
            conversational_response = await self.get_conversational_response(user_input)
            if conversational_response:
                return conversational_response
        
        response_content = await self.process_user_input(channel, user_input)
        await self.clear_status()
        return response_content

    async def is_conversational_prompt(self, user_input):
        """Determine if the user input is a conversational prompt or a command using Mistral AI."""
        try:
            messages = [
                {"role": "system", "content": """You are an AI assistant that determines if a user's message is a conversational prompt or a command.
                Analyze the input and respond with a JSON object:
                {
                    "is_conversational": boolean,
                    "type": string (one of: "greeting", "introduction", "help_request", "command")
                }
                
                Examples of conversational prompts:
                - Greetings (hi, hello, hey)
                - Questions about the assistant (who are you, what can you do)
                - Help requests (help, can you help me)
                
                Everything else should be considered a command."""},
                {"role": "user", "content": f"Analyze this input: {user_input}"}
            ]
            
            response = await self.mistral_client.chat.complete_async(
                model=MISTRAL_MODEL,
                messages=messages
            )
            
            if response and response.choices:
                result = json.loads(response.choices[0].message.content)
                return result["is_conversational"]
            
            return False
        except:
            # Fallback to basic pattern matching if Mistral fails
            conversational_keywords = ['hello', 'hi', 'who are you', 'what can you do', 'help']
            return any(keyword in user_input.lower() for keyword in conversational_keywords)

    async def get_conversational_response(self, user_input):
        """Generate natural conversational responses using Mistral AI."""
        try:
            messages = [
                {"role": "system", "content": """You are an accessibility assistant having a conversation with someone.
                Respond in a warm, engaging way that matches the user's conversational tone.
                
                Key points about yourself:
                - You help users with vision impairments navigate websites
                - You can describe web pages, images, and content
                - You can help navigate through links and sections
                - You're focused on making web content accessible
                
                Keep responses natural and conversational. Always end with an invitation to explore websites or ask questions.
                Avoid bullet points or numbered lists unless specifically asked for help commands."""},
                {"role": "user", "content": user_input}
            ]
            
            response = await self.mistral_client.chat.complete_async(
                model=MISTRAL_MODEL,
                messages=messages
            )
            
            if response and response.choices:
                return response.choices[0].message.content
            
            # Fallback responses if Mistral fails
            if "who are you" in user_input.lower():
                return ("I'm an accessibility assistant designed to help users with vision impairments navigate websites. "
                        "I can help you explore web pages, describe their content, navigate through links, and understand images. "
                        "Would you like to visit a website? Just share the URL with me, and I'll guide you through it!")
            
            if any(greeting in user_input.lower() for greeting in ["hello", "hi", "hey"]):
                return ("Hello! I'm your accessibility assistant. I can help you navigate websites and understand their content. "
                        "To get started, just share a website URL with me, and I'll help you explore it!")
            
            if "what can you do" in user_input.lower() or "help" in user_input.lower():
                return ("I can help you explore websites by describing their content, images, and helping you navigate through different sections. "
                        "To get started, just share a website URL with me!")
            
            return None
        except Exception as e:
            # Fallback to basic responses if something goes wrong
            return ("Hello! I'm your accessibility assistant. I can help you navigate websites and understand their content. "
                    "To get started, just share a website URL with me!")

    async def get_compact_page_description(self, context):
        """Generate a conversational description of the page."""
        try:
            # Prepare the prompt for the main content
            messages = [
                {"role": "system", "content": """You are accessibility assistant having a conversation with someone who cannot see the website. 
                Describe what you see in a detailed yet natural way using daily language.

                When describing the page:
                1. Start with a comprehensive overview of what the page is about and its main purpose
                2. Describe the layout and organization of the content in a natural way
                3. Mention 3-4 important features, sections, or interactive elements
                4. Point out any notable images, media, or visual content that adds value
                5. Highlight any unique or special features that make this page interesting
                6. Suggest natural next steps or areas worth exploring

                Aim for descriptions that paint a clear picture while maintaining a conversational tone.
                Include enough specifics to help users understand the full scope of what's available,
                but present it in a flowing, natural way rather than as a list.
                
                Let users know what options they have without overwhelming them."""},
                                {"role": "user", "content": f"""Describe this webpage in a natural, detailed way covering:
                1. What is this page about and what's its main purpose?
                2. How is the content organized and laid out?
                3. What are the 3-4 most important things someone can do or find here?
                4. Are there any notable images, media, or visual elements worth describing?
                5. What makes this page unique or interesting?
                6. What would be most helpful to explore first?

                Content: {json.dumps(context)}"""}
            ]

            response = await self.mistral_client.chat.complete_async(
                model=MISTRAL_MODEL,
                messages=messages
            )

            if response and response.choices and len(response.choices) > 0:
                main_description = response.choices[0].message.content
                
                # Add a natural transition to available actions
                final_response = main_description + "\n\n"
                final_response += "I can help you explore more - just let me know what interests you most. "
                final_response += "You can ask about specific sections, images, or click on any link you'd like to explore."
                
                return final_response
                
            return "I'm having trouble understanding this page. Would you like me to try again?"
        except Exception as e:
            return f"I ran into an issue while looking at this page: {str(e)}. How can I help you?"

    def __del__(self):
        """Clean up WebDriver when the agent is destroyed"""
        if hasattr(self, 'driver'):
            self.driver.quit()