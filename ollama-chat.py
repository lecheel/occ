import json
import sys
import os
import subprocess
import threading
import paho.mqtt.client as mqtt
import argparse
import re
from ollama import Client, AsyncClient
from typing import Dict, Optional
from datetime import datetime
from colorama import init, Fore, Style
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import clear
from chat_config import CHAT_SYSTEM_PROMPT, CODE_SYSTEM_PROMPT
import logging
from logging.handlers import RotatingFileHandler
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import asyncio
import random
# Initialize colorama
init()

# Available colors for the logo
LOGO_COLORS = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]

# Ollama Logo
logos = ["""\
@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@#*@@@@@@+#@@@@@@@
@@@@@@+:+:%##%:+:+@@@@@@
@@@@@@--* =**= *:-@@@@@@
@@@@@*:+#%@@@@%#+:*@@@@@
@@@@% @@*@#**#@*@@ @@@@@
@@@@@:+%*:#**#:*%+:@@@@@
@@@@@.*@@******@@*.@@@@@
@@@@@.#@@@@@@@@@@#.@@@@@
@@@@@--@@@@@@@@@@--@@@@@
@@@@@.#@@@@@@@@@@*.@@@@@""",
"""\
@@@@@@@@@@@@=*+*@@@@@@@@
@@@@@*#%@%##=%@*+%@@@@@@
@@@@@-*=*#%@@@@@@*=@@@@@
@@@@@@-@@@@%#%@+%@=*@@@@
@@@@@@+*+++*#*-*#@%=%@@@
@@@@@@%=%*+**+*@@@@=*@@@
***+%@@=%@@%%@@@@@@#=%@@
#**+#%#%-#%@@@@@@@@@-#@@
%###=#=+@@%%*%@@%%@@*+@@
#***%@@*+*@@@+*+#@@%%:@@
#%@@@@@@#+%%@%==###%####
@@@@@@@@@@%%#%##@@@@@@@@""",        
]

def render_logo():
    # just print random logo 
    logo_index = random.randint(0, len(logos) - 1)
    logo = logos[logo_index]
    print(logo)

class MQTTHandler:
    def __init__(self, chat_instance):
        self.client = mqtt.Client(protocol=mqtt.MQTTv5)
        self.chat = chat_instance
        
        # Setup callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        
        # Default MQTT settings
        self.broker = "localhost"
        self.port = 1883
        self.topic_sub = "ollama/command/#"
        self.topic_pub = "ollama/response"
        
        # Create event loop for MQTT handler
        self.loop = asyncio.new_event_loop()
        
        # Start MQTT thread
        self.mqtt_thread = threading.Thread(target=self.run, daemon=True)
        self.mqtt_thread.start()
    
    def run(self):
        """Run the MQTT client loop."""
        try:
            # Connect to broker
            self.client.connect(self.broker, self.port)
            
            # Set the event loop for this thread
            asyncio.set_event_loop(self.loop)
            
            # Start the MQTT loop
            self.client.loop_forever()
            
        except Exception as e:
            print(f"{Fore.RED}MQTT Error: {str(e)}{Style.RESET_ALL}")
    
    def on_connect(self, client, userdata, flags, rc, properties=None):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            # print(f"{Fore.GREEN}Connected to MQTT broker{Style.RESET_ALL}")
            self.client.subscribe(self.topic_sub)
        else:
            print(f"{Fore.RED}Failed to connect to MQTT broker: {rc}{Style.RESET_ALL}")
    
    def on_disconnect(self, client, userdata, rc, properties=None):
        """Callback when disconnected from MQTT broker."""
        print(f"{Fore.YELLOW}Disconnected from MQTT broker{Style.RESET_ALL}")

    def on_message(self, client, userdata, msg):
        """Handle incoming MQTT messages."""
        try:
            command = msg.payload.decode()
            print(f"\nReceived command on {msg.topic}: {command}")
            
            # Generate response using async client
            response = self.loop.run_until_complete(self.chat.generate_response_async(command))
            
            # Publish response
            self.publish_response(response)
            
        except Exception as e:
            error_msg = f"Error processing command: {str(e)}"
            print(f"\n{Fore.RED}{error_msg}{Style.RESET_ALL}")
            self.publish_response(error_msg)

    def publish_response(self, message: str):
        """Publish response to MQTT topic."""
        try:
            self.client.publish(self.topic_pub, message)
        except Exception as e:
            print(f"{Fore.RED}Error publishing to MQTT: {str(e)}{Style.RESET_ALL}")

class OllamaChat:
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434", word_output: bool = False, debug: bool = False):
        """
        Initialize the Ollama chat wrapper.
        
        Args:
            model (str): The model to use for chat (default: "llama2")
            base_url (str): Base URL for Ollama API (default: "http://localhost:11434")
            word_output (bool): Save responses (excluding code blocks) to word.txt (default: False)
            debug (bool): Enable debug output (default: False)
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.conversation_history = []
        self.system_prompt = CHAT_SYSTEM_PROMPT
        self.last_response = ""
        self.word_output = word_output
        
        # Initialize Ollama client
        self.client = Client(host=self.base_url)
        self.async_client = AsyncClient(host=self.base_url)
        
        # Setup logging
        self.logger = logging.getLogger('OllamaChat')
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'ollama_chat.log')
        
        # Create file handler
        file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
        
        # Create formatter and add it to handler
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Add handler to the logger
        self.logger.addHandler(file_handler)
        
        # Prevent logs from propagating to the root logger (which would show on console)
        self.logger.propagate = False
        
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        # Initialize commands
        self.commands = {
            '/help': self.show_help,
            '/clear': self.clear_history,
            '/save': self.save_conversation,
            '/load': self.load_conversation,
            '/system': self.set_system_prompt,
            '/model': self.change_model,
            '/status': self.show_status,
            '/ja': self.load_mic_context,
            '/jc': self.load_mic_content,
            '/cb': self.show_code_blocks,
        }

        # Setup prompt_toolkit session
        history_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.chat_history')
        words = load_words()
        completion_words = list(self.commands.keys()) + words
        self.command_completer = WordCompleter(
            completion_words,
            ignore_case=True,
            match_middle=True
        )

        self.session = PromptSession(
            history=FileHistory(history_file),
            completer=self.command_completer,
            complete_while_typing=True
        )
        
        # Initialize MQTT handler
        self.mqtt_handler = MQTTHandler(self)
        
    def run(self):
        """Main chat loop."""
        user_cmd = False
        
        # Print logo and welcome message
        render_logo()
        print(f"{Fore.GREEN}Chat started. Type /help for available commands. Press Ctrl+D to exit.{Style.RESET_ALL}")
        print(f"Using model: {self.model}\n")
        
        try:
            # Create event loop for the chat session
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            while True:
                try:
                    # Reset command flag at start of loop
                    user_cmd = False
                    
                    # Get input using prompt_toolkit
                    user_input = self.session.prompt("💭 : ")
                    
                    # Skip empty input
                    if not user_input.strip():
                        continue
                    
                    # Check for exit commands
                    if user_input.lower() in ['exit', 'quit', 'q']:
                        user_cmd = True
                        break
                    if user_input.lower() in ['cls']:
                        user_cmd = True
                        clear()
                    
                    # Handle commands
                    if user_input.startswith('/'):
                        user_cmd = True
                        command = user_input.split()[0]
                        args = user_input[len(command):].strip()
                        
                        if command in self.commands:
                            self.commands[command](args)
                            continue
                        else:
                            print(f"{Fore.RED}Unknown command: {command}{Style.RESET_ALL}")
                            continue
                    
                    # Skip response generation if this was a command
                    if user_cmd:
                        continue
                    
                    # Generate and display response
                    print(f"\n󰮯 : ", end="", flush=True)
                    response = loop.run_until_complete(self.generate_response_async(user_input))
                    self.last_response = response
                    sys.stdout.flush()
                    
                    # Add to conversation history
                    self.conversation_history.extend([
                        {"role": "user", "content": user_input},
                        {"role": "assistant", "content": response}
                    ])
                    
                    # Save to word.txt if enabled
                    if self.word_output:
                        self.save_to_word(response)
                        
                except KeyboardInterrupt:
                    print("\nInterrupted by user")
                    break
                except EOFError:
                    break
                except Exception as e:
                    self.handle_error(f"Error in chat loop: {str(e)}")
                    if self.logger.level == logging.DEBUG:
                        import traceback
                        self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
                    
        except Exception as e:
            self.handle_error(f"Fatal error in chat: {str(e)}")
        finally:
            # Clean up the event loop
            try:
                loop.close()
            except:
                pass
            print(f"\n{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
        
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the conversation."""
        self.system_prompt = prompt

    def save_to_word(self, text: str) -> None:
        """Save response text (excluding code blocks) to word.txt."""
        try:
            # Remove code blocks
            text_without_code = re.sub(r'```[\s\S]*?```', '', text)
            
            # Remove remaining backticks and clean up
            text_without_code = re.sub(r'`[^`]*`', '', text_without_code)
            
            # Clean up extra newlines
            text_without_code = re.sub(r'\n\s*\n', '\n\n', text_without_code.strip())
            
            # Get timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Append to word.txt
            with open('word.txt', 'a', encoding='utf-8') as f:
                f.write(f"\n--- {timestamp} ---\n")
                f.write(text_without_code)
                f.write("\n\n")
                
        except Exception as e:
            self.handle_error(f"Error saving to word.txt: {str(e)}")

    async def generate_response_async(self, prompt: str) -> str:
        """
        Generate a response from Ollama using async streaming.
        
        Args:
            prompt (str): User input prompt
            
        Returns:
            str: Generated response
        """
        try:
            full_response = ""
            print(f"\n󰮯 : ", end="", flush=True)
            
            # Create messages list
            messages = [{"role": "system", "content": self.system_prompt}]
            if self.conversation_history:
                self.logger.debug(f"Adding conversation history: {len(self.conversation_history)} messages")
                messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": prompt})
            
            self.logger.debug(f"Final messages structure: {json.dumps(messages, indent=2)}")
            self.logger.debug(f"Using model: {self.model}")
            
            # Stream the response using async client
            async for chunk in await self.async_client.chat(
                model=self.model,
                messages=messages,
                stream=True,
            ):
                self.logger.debug(f"Raw chunk received: {chunk}")
                if chunk and 'message' in chunk:
                    response_text = chunk['message']['content']
                    full_response += response_text
                    print(response_text, end='', flush=True)
                else:
                    self.logger.warning(f"Received chunk without message: {chunk}")
            
            if not full_response:
                self.logger.warning("No response text generated")
            else:
                self.logger.debug(f"Final response length: {len(full_response)}")
            
            print()  # Add a newline after streaming
            # Store the response for code block extraction
            self.last_response = full_response
            # Save to word.txt if enabled
            if self.word_output:
                self.save_to_word(full_response)
            # Print formatted version of the response
            render_markdown(full_response)
            print()
            return full_response
            
        except Exception as e:
            self.handle_error(f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _save_debug_state(self, state_type: str, data: dict):
        """Save debug state to a file for later analysis."""
        try:
            debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_dumps')
            os.makedirs(debug_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(debug_dir, f'debug_state_{state_type}_{timestamp}.json')
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"Saved debug state to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save debug state: {e}")

    def save_conversation(self, filename: Optional[str] = None) -> None:
        """
        Save the conversation history to a file.
        
        Args:
            filename (str, optional): Custom filename, defaults to timestamp
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
        
        data = {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "conversation": self.conversation_history
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\nConversation saved to {filename}")
        except IOError as e:
            self.handle_error(f"Error saving conversation: {e}")

    def load_conversation(self, filename: str) -> None:
        """
        Load a conversation history from a file.
        
        Args:
            filename (str): Path to the conversation file
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.model = data.get('model', self.model)
                self.system_prompt = data.get('system_prompt', self.system_prompt)
                self.conversation_history = data.get('conversation', [])
            print(f"\nConversation loaded from {filename}")
        except IOError as e:
            self.handle_error(f"Error loading conversation: {e}")
    
    def show_help(self, args: str = ""):
        """Show available commands and their usage."""
        help_text = f"""
{Fore.CYAN}Available Commands:{Style.RESET_ALL}
{Fore.YELLOW}/help{Style.RESET_ALL} - Show this help message
{Fore.YELLOW}/clear{Style.RESET_ALL} - Clear conversation history
{Fore.YELLOW}/save [filename]{Style.RESET_ALL} - Save conversation history
{Fore.YELLOW}/load <filename>{Style.RESET_ALL} - Load conversation history
{Fore.YELLOW}/system <prompt>{Style.RESET_ALL} - Set system prompt
{Fore.YELLOW}/model [model_name]{Style.RESET_ALL} - Change the model
{Fore.YELLOW}/status{Style.RESET_ALL} - Show current AI settings
{Fore.YELLOW}/ja{Style.RESET_ALL} - Load context from mic.txt
{Fore.YELLOW}/jc{Style.RESET_ALL} - Send content from mic.txt as chat message
{Fore.YELLOW}/cb{Style.RESET_ALL} - Show or save code blocks from last response
{Fore.YELLOW}exit{Style.RESET_ALL} or {Fore.YELLOW}quit{Style.RESET_ALL},{Fore.YELLOW}q{Style.RESET_ALL} - End the chat session
"""
        print(help_text)

    def clear_history(self, args: str = ""):
        """Clear the conversation history."""
        self.conversation_history = []
        print(f"{Fore.YELLOW}Conversation history cleared.{Style.RESET_ALL}")

    def change_model(self, args: str):
        """Change the model being used."""
        if not args:
            print(f"\nCurrent model: {self.model}")
            try:
                models = self.client.list()
                print("\nAvailable models:")
                for model in models:
                    print(f"- {model['name']}")
            except Exception as e:
                self.handle_error(f"Error listing models: {str(e)}")
            return

        new_model = args.strip()
        self.model = new_model
        print(f"\nSwitched to model: {self.model}")
        
    def load_mic_context(self, args: str = "") -> None:
        """Load context from mic.txt and use it in the conversation."""
        try:
            mic_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mic.txt')
            if not os.path.exists(mic_file):
                print(f"{Fore.RED}Error: mic.txt not found in the current directory.{Style.RESET_ALL}")
                return

            with open(mic_file, 'r', encoding='utf-8') as f:
                context = f.read().strip()

            if not context:
                print(f"{Fore.RED}Error: mic.txt is empty.{Style.RESET_ALL}")
                return

            # Add the context to the conversation history
            self.conversation_history.append({
                "role": "system",
                "content": f"Additional context: {context}"
            })
            print(f"{Fore.YELLOW}Context loaded from mic.txt:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{context}{Style.RESET_ALL}")

        except Exception as e:
            self.handle_error(f"Error loading mic.txt: {str(e)}")

    def load_mic_content(self, args: str = "") -> None:
        """Load content from mic.txt and send it directly as a chat message."""
        try:
            mic_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mic.txt')
            if not os.path.exists(mic_file):
                print(f"{Fore.RED}Error: mic.txt not found in the current directory.{Style.RESET_ALL}")
                return

            with open(mic_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                print(f"{Fore.RED}Error: mic.txt is empty.{Style.RESET_ALL}")
                return

            print(f"{Fore.YELLOW}Sending content from mic.txt:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{content}{Style.RESET_ALL}")
            
            # Generate response for the content
            print(f"{Fore.BLUE}Assistant: {Style.RESET_ALL}", end='')
            response = asyncio.run(self.generate_response_async(content))
            print()  # Add a newline after the response

        except Exception as e:
            self.handle_error(f"Error loading mic.txt: {str(e)}")

    def extract_code_blocks(self, text: str) -> list:
        """Extract code blocks from markdown text."""
        code_blocks = []
        
        # Regex to capture language and code
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for lang, code in matches:
            # Preserve the literal \n in string literals but handle actual newlines
            code = code.replace('\\n', '\x00').replace('\n', '\n').replace('\x00', '\\n')
            # Append as a dictionary with language and code
            code_blocks.append({
                'language': lang.strip() if lang else 'plaintext',  # Default to 'plaintext' if no language specified
                'code': code.strip()  # Trim any leading/trailing whitespace
            })

        return code_blocks

    def show_code_blocks(self, args: str = ""):
        """Show or save code blocks from the last response.
        Usage: 
            /cb           - Show all code blocks
            /cb [n]      - Show specific code block number
            /cb file.py  - Save first code block to file.py
            /cb file.py [n] - Save specific code block number to file.py
        """
        code_blocks = self.extract_code_blocks(self.last_response)
        
        if not code_blocks:
            print(f"{Fore.RED}No code blocks found.{Style.RESET_ALL}")
            return
            
        # Parse arguments
        args = args.strip().split()
        if not args:
            # Show all code blocks
            for i, block in enumerate(code_blocks, start=1):
                lang = block.get('language', 'plain')  # Use 'plain' as default if no language specified
                code = block.get('code', '')  # Get the code, default to empty string if not found
                print(f"\n{Fore.CYAN}Code Block {i} ({lang}):{Style.RESET_ALL}")
                print(code)
            return
            
        try:
            if len(args) == 1:
                # Check if it's a number or filename
                if args[0].isdigit():
                    # Show specific block
                    block_num = int(args[0])
                    if 1 <= block_num <= len(code_blocks):
                        block = code_blocks[block_num - 1]
                        lang = block.get('language', 'plain')  # Use 'plain' as default if no language specified
                        code = block.get('code', '')  # Get the code, default to empty string if not found
                        print(f"\n{Fore.CYAN}Code Block {block_num} ({lang}):{Style.RESET_ALL}")
                        print(code)
                    else:
                        print(f"{Fore.RED}Invalid code block number. Available blocks: 1-{len(code_blocks)}{Style.RESET_ALL}")
                else:
                    # Save first block to file
                    filename = args[0]
                    block = code_blocks[0]
                    self._save_code_block(filename, block.get('code', ''))
            
            elif len(args) == 2:
                # Save specific block to file
                filename, block_num = args[0], args[1]
                if not block_num.isdigit():
                    print(f"{Fore.RED}Invalid syntax. Use: /cb filename.py [block_number]{Style.RESET_ALL}")
                    return
                    
                block_num = int(block_num)
                if 1 <= block_num <= len(code_blocks):
                    block = code_blocks[block_num - 1]
                    self._save_code_block(filename, block.get('code', ''))
                else:
                    print(f"{Fore.RED}Invalid code block number. Available blocks: 1-{len(code_blocks)}{Style.RESET_ALL}")
            
            else:
                print(f"{Fore.RED}Invalid syntax. Use: /cb [filename.py] [block_number]{Style.RESET_ALL}")
                
        except Exception as e:
            self.handle_error(f"Error handling code blocks: {str(e)}")
            
    def _save_code_block(self, filename: str, code: str):
        """Helper method to save code block to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            with open(filename, 'w') as f:
                f.write(code)
            print(f"{Fore.GREEN}Code block saved to: {filename}{Style.RESET_ALL}")
            
        except Exception as e:
            self.handle_error(f"Error saving code block to {filename}: {str(e)}")

    def show_status(self, args: str = ""):
        """Show current AI model settings and parameters."""
        try:
            print(f"\n{Fore.CYAN}Current Settings:{Style.RESET_ALL}")
            print(f"Model: {self.model}")
            print(f"API URL: {self.base_url}")
            
            # Get model details
            try:
                model_info = self.client.show(model=self.model)
                self.logger.debug(f"Raw model info: {model_info}")
                
                if isinstance(model_info, dict):
                    # Print model parameters
                    print(f"\n{Fore.CYAN}Model Parameters:{Style.RESET_ALL}")
                    parameters = model_info.get('parameters', {})
                    if isinstance(parameters, dict):
                        for key, value in parameters.items():
                            print(f"{key}: {value}")
                    else:
                        print("Parameters:", parameters)
                    
                    # Print model details
                    template = model_info.get('template', '')
                    if template:
                        print(f"\n{Fore.CYAN}Chat Template:{Style.RESET_ALL}")
                        print(template)
                    
                    # Print model license and size
                    print(f"\n{Fore.CYAN}Model Details:{Style.RESET_ALL}")
                    for key in ['size', 'format', 'family']:
                        value = model_info.get(key, 'N/A')
                        if value != 'N/A':
                            print(f"{key}: {value}")
                else:
                    print(f"\n{Fore.CYAN}Model Info:{Style.RESET_ALL}")
                    print(model_info)
                
            except Exception as e:
                self.logger.error(f"Error getting model details: {str(e)}")
                self.logger.debug("Attempting to get model list instead...")
                
                # Try to get model info another way
                try:
                    models = self.client.list()
                    self.logger.debug(f"Raw models list: {models}")
                    
                    if isinstance(models, list):
                        for model in models:
                            if isinstance(model, dict) and model.get('name') == self.model:
                                print(f"\n{Fore.CYAN}Model Details from list:{Style.RESET_ALL}")
                                for key, value in model.items():
                                    print(f"{key}: {value}")
                                break
                    else:
                        print(f"\n{Fore.CYAN}Available Models:{Style.RESET_ALL}")
                        print(models)
                        
                except Exception as e2:
                    self.logger.error(f"Error getting model list: {str(e2)}")
            
            # Print system prompt
            print(f"\n{Fore.CYAN}Current System Prompt:{Style.RESET_ALL}")
            print(self.system_prompt)
            
            # Print conversation stats
            print(f"\n{Fore.CYAN}Conversation Stats:{Style.RESET_ALL}")
            print(f"Messages in history: {len(self.conversation_history)}")
            
        except Exception as e:
            self.handle_error(f"Error showing status: {str(e)}")
            
    def handle_error(self, error_msg: str):
        """Handle errors in a consistent way."""
        self.logger.error(error_msg)
        print(f"{Fore.RED}{error_msg}{Style.RESET_ALL}")

def load_words(filename: str = "word_clean.txt") -> list[str]:
    # check if the file exists
    homedir=os.path.expanduser("~")
    filename = os.path.join(homedir, filename)
    
    """Load words from the clean word file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            # Skip header lines starting with #
            words = [line.strip() for line in f if not line.startswith('#') and line.strip()]
        return words
    except FileNotFoundError:
        print(f"{Fore.RED}Error: '{filename}' not found{Style.RESET_ALL}")
        return []
    except Exception as e:
        print(f"{Fore.RED}Error loading words: {str(e)}{Style.RESET_ALL}")
        return []

def render_markdown(text: str) -> None:
    """Render markdown text using mdcat."""
    try:
        # Create a temporary file to store the markdown
        temp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.temp_response.md')
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Use mdcat to render the markdown
        subprocess.run(['mdcat', temp_file], check=True)
        
        # Clean up temporary file
        os.remove(temp_file)
    except subprocess.CalledProcessError:
        # Fallback to plain text if mdcat fails
        print(text)
    except Exception as e:
        print(f"{Fore.RED}Error rendering markdown: {str(e)}{Style.RESET_ALL}")
        print(text)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Ollama Chat CLI with advanced features')
    parser.add_argument('--word', action='store_true', help='Save responses (excluding code blocks) to word.txt')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Create and start the chat
    chat = OllamaChat(word_output=args.word, debug=args.debug)
    chat.run()
