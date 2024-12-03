"""Configuration settings for the Ollama chat interface."""

# Chat system configurations
CHAT_SYSTEM_PROMPT = """You are a general AI assistant.

The user provided the additional info about how they would like you to respond:

- If you're unsure don't guess and say you don't know instead.
- Ask question if you need clarification to provide better answer.
- Think deeply and carefully from first principles step by step.
- Zoom out first to see the big picture and then zoom in to details.
- Use Socratic method to improve your thinking and coding skills.
- Don't elide any code from your output if the answer requires coding.
- Take a deep breath; You've got this!
"""

CODE_SYSTEM_PROMPT = """You are an AI working as a code editor.

Please AVOID COMMENTARY OUTSIDE OF THE SNIPPET RESPONSE.
START AND END YOUR ANSWER WITH:

```"""

# Chat template configurations
CHAT_TEMPLATE = """# topic: ?

- file: {filename}
{optional_headers}
Write your queries after {user_prefix}. Use `{respond_shortcut}` or :{cmd_prefix}ChatRespond to generate a response.
Response generation can be terminated by using `{stop_shortcut}` or :{cmd_prefix}ChatStop command.
Chats are saved automatically. To delete this chat, use `{delete_shortcut}` or :{cmd_prefix}ChatDelete.
Be cautious of very long chats. Start a fresh chat by using `{new_shortcut}` or :{cmd_prefix}ChatNew.

---

{user_prefix}
"""

SHORT_CHAT_TEMPLATE = """# topic: ?
- file: {filename}
---

{user_prefix}
"""
