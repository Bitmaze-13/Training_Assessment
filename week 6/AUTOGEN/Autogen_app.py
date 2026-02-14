from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent,UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from pathlib import Path
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
import os
import asyncio

load_dotenv(dotenv_path="D:\GenAI_Langchain\\tchat\.env")

model_info = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
    "family": "unknown",
    "structured_output": True
}

llm = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model_info=model_info
    
)

assistant = AssistantAgent(
    name="Assistant",
    model_client=llm,
    description="You're an Senior AI engineer who only give runnable python code."
)

code_executor = CodeExecutorAgent(
    name= "ExecutorAgent",
    model_client=llm,
    code_executor=LocalCommandLineCodeExecutor(work_dir=Path.cwd() / "runs")
)

termination = TextMentionTermination(
    "exit", sources=["user"]
)

user = UserProxyAgent(name="user")

team = RoundRobinGroupChat(
    participants=[user, assistant, code_executor],
    termination_condition=termination
)

# 4. Async Execution Wrapper
async def main():
    try:
        await Console(team.run_stream())
    finally:
        llm.close()


if __name__ == "__main__":
    asyncio.run(main())