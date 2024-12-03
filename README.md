# Ollama Chat CLI

An advanced command-line interface for interacting with Ollama AI models, featuring async streaming responses, MQTT integration, and a modern prompt interface.

## Features

- 🚀 Asynchronous streaming responses
- 🎨 Beautiful colored terminal output with Ollama logo
- 📝 Markdown rendering with `mdcat`
- 🔍 Code block extraction and management
- 🤖 Multiple model support with dynamic switching
- 💾 Conversation history management
- 🔄 MQTT pub/sub integration with async support
- ⌨️ Modern prompt interface with auto-completion
- 📜 Command history with file persistence
- 🎯 Tab completion for commands and models
- 🐛 Comprehensive error handling and debugging
- 📊 Status command for model information

## Dependencies

- Python 3.10+
- Ollama server running locally
- Required Python packages:
  ```
  ollama
  colorama
  prompt_toolkit
  paho-mqtt
  ```
- External tools:
  - `mdcat` for markdown rendering
  - MQTT broker (for pub/sub features)

## Installation

1. Install required Python packages:
   ```bash
   pip install ollama-python colorama prompt_toolkit paho-mqtt
   ```

2. Install mdcat (for markdown rendering):
   ```bash
   # On Ubuntu/Debian
   apt install mdcat
   # On macOS
   brew install mdcat
   ```

3. Install and start Ollama server:
   Follow instructions at [Ollama's official website](https://ollama.ai)

4. (Optional) Install MQTT broker:
   ```bash
   # On Ubuntu/Debian
   apt install mosquitto
   ```

## Usage

Run the script:
```bash
python ollama-chat.py [--word] [--debug]
```

Options:
- `--word`: Save responses (excluding code blocks) to word.txt
- `--debug`: Enable debug logging

### Available Commands

- `/help` - Show available commands
- `/clear` - Clear conversation history
- `/save` - Save conversation to file
- `/load` - Load conversation from file
- `/system` - Set system prompt
- `/model` - Change AI model
- `/status` - Show model information and chat status
- `/ja` - Load context from mic.txt
- `/jc` - Send mic.txt content as chat message
- `/cb` - Extract and display code blocks from last response

### Exit Methods
- Type `exit`, `quit`, or `q`
- Press `Ctrl+C` to interrupt
- Press `Ctrl+D` to send EOF signal

### MQTT Integration

Topics:
- `ollama/command` - Send chat messages
- `ollama/command/system` - Update system prompt
- `ollama/command/model` - Change model
- `ollama/response` - Receive responses

## Features in Detail

### Async Response Streaming
- Real-time token-by-token output
- Non-blocking response generation
- Proper event loop management
- Rate limit handling

### Modern Prompt Interface
- Real-time command completion
- History navigation with up/down arrows
- Multi-line editing support
- Smart tab completion for commands and models

### MQTT Integration
- Async publish/subscribe messaging
- Remote command execution
- Dedicated event loop for MQTT handler
- Multiple client support

### Code Block Handling
- Automatic code block extraction
- Language detection
- Syntax highlighting (when available)
- Save blocks to files with `/cb` command

### Error Handling
- Comprehensive error catching
- Debug mode with full tracebacks
- Graceful exit handling
- Rate limit management

### Logging
- Rotating file logs
- Debug mode for detailed information
- Error tracking and reporting
- MQTT message logging

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
