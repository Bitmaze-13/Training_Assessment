import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import UserProxyAgent, AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from dotenv import load_dotenv
import sys

load_dotenv(dotenv_path="D:\GenAI_Langchain\\tchat\.env")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

model_info = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True
}


gemini_client = OpenAIChatCompletionClient(
 model="gemini-3-flash-preview",
 base_url="https://generativelanguage.googleapis.com/v1beta/openai/", 
 model_info=model_info

)

async def main() -> None:
    web_surfer_agent = MultimodalWebSurfer(
        name="web_surf",
        model_client=gemini_client
    )
    user = AssistantAgent(name="user",
                          model_client=gemini_client,
                          description="you're an ai chatbot that summarize the web information in structured format")

    at = RoundRobinGroupChat(
        participants=[web_surfer_agent, user],
        max_turns=3
    )

    stream = at.run_stream(task="Navigate to Google and search about Abid Ali Awan.")
    await Console(stream)
    await web_surfer_agent.close()

    print("Agent initialized successfully!")

if __name__ =="__main__":
    asyncio.run(main())