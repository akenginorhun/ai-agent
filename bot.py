import os
import discord
import logging
import re

from discord.ext import commands
from dotenv import load_dotenv
from agent import AccessibilityAgent

PREFIX = "!"

# Setup logging
logger = logging.getLogger("discord")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# Load the environment variables
load_dotenv()

# Create the bot with all intents
# The message content and members intent must be enabled in the Discord Developer Portal for the bot to work.
intents = discord.Intents.all()
bot = commands.Bot(command_prefix=PREFIX, intents=intents)

# Import the Accessibility agent
agent = AccessibilityAgent()

# Get the token from the environment variables
token = os.getenv("DISCORD_TOKEN")

@bot.event
async def on_ready():
    """
    Called when the client is done preparing the data received from Discord.
    Prints message on terminal when bot successfully connects to discord.

    https://discordpy.readthedocs.io/en/latest/api.html#discord.on_ready
    """
    logger.info(f"{bot.user} has connected to Discord!")
    logger.info("Accessibility Assistant is ready to help!")

@bot.event
async def on_message(message: discord.Message):
    """Process messages and help users navigate websites accessibly."""
    # Don't delete this line! It's necessary for the bot to process commands.
    await bot.process_commands(message)

    # Ignore messages from self or other bots to prevent infinite loops
    if message.author.bot or message.content.startswith(PREFIX):
        return

    # Process the message with the accessibility agent
    logger.info(f"Processing message from {message.author}: {message.content}")
    
    try:
        # Add typing indicator to show the bot is processing
        async with message.channel.typing():
            # Get the response from the agent
            response = await agent.run(message)

            if not response:
                await message.channel.send("I'm not sure how to help with that. Try asking me to visit a website, describe something specific, or use the !guide command for help.")
                return

            # Split long messages if needed (Discord has a 2000 character limit)
            if len(response) > 1900:
                parts = [response[i:i+1900] for i in range(0, len(response), 1900)]
                for part in parts:
                    await message.channel.send(part)
            else:
                await message.channel.send(response)
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        error_message = (
            "I encountered an error while processing your request. Here are some things you can try:\n"
            "1. Make sure the URL is valid and accessible\n"
            "2. Try being more specific with your request\n"
            "3. Use the !guide command to see how to use me effectively\n"
            "4. If the problem persists, try a different website or action"
        )
        await message.channel.send(error_message)

# Commands

# This example command is here to show you how to add commands to the bot.
# Run !ping with any number of arguments to see the command in action.
# Feel free to delete this if your project will not need commands.
@bot.command(name="guide", help="Shows how to use the accessibility assistant.")
async def guide_command(ctx):
    help_text = """
**🌐 Accessibility Assistant Guide**

I can help you navigate websites and understand their content! Here's how to use me:

1. **Send a URL** - I'll visit the website and describe its content to you
2. **Navigate** - Tell me what you'd like to do on the page:
   - "Click on [link name]"
   - "Read more about [topic]"
   - "Describe the images"
   - "Go back"

3. **Ask Questions** - I can help you understand:
   - What's on the page
   - Where things are located
   - What actions you can take

Just send me a website URL or ask me about the current page!
"""
    await ctx.send(help_text)

# Start the bot, connecting it to the gateway
bot.run(token)
