from crewai import Agent, LLM, Task, Crew
from crewai_tools import SerperDevTool, PDFSearchTool, SerpApiGoogleSearchTool
import os
# import ssl
from dotenv import load_dotenv
load_dotenv()
azure_llm = LLM(
    # model="gemini-2.5-flash"
    model="azure/gpt-4o-2"
)
# researcher = Agent(
#     role= "AI Researcher",
#     goal="Gather the latest AI trends from 2024",
#     backstory="You are an expert Ai researcher Who writes reports on emerging Ai trends.",
#     verbose=True,
#     llm = azure_llm
# )

search = SerpApiGoogleSearchTool()

# Research_task = Task(
#     description= "Identify the top 3 Ai trends in 2024 using online resource.",
#     expected_output="a list of of the top 3 trends with a short summary of each",
#     agent=researcher,
#     verbose=True

# )



# writer = Agent(
#     role= "Expert Author",
#     goal="Write a Blogs for AI enthusiasts.",
#     backstory= "Experienced Blogger writing Professional and engaging blogs with more than 30+ years experience.",
#     llm = azure_llm,
# )

# Write_task = Task(
#     description="Write a blog post based on the findings.",
#     expected_output="a 300-word post",
#     agent=writer,
#     context=[Research_task]
# )

# Ai_agent = Crew(
#     agents=[writer],
#     tasks=[Write_task],
#     verbose=True
# )


# os.environ["OPENAI_API_KEY"] = "NA"
# #tools crewai
# pdf_tool = PDFSearchTool(
#     pdf="Ai.pdf",
#     config=dict(
#         llm=dict(
#             provider="azure_openai",
#             config=dict(
#                 model="azure/gpt-4o-2", # Must match your Azure deployment
#                 # Note: CrewAI usually pulls AZURE_OPENAI_API_KEY from env automatically
#             ),
#         ),
#         embedder=dict(
#             provider="azure_openai",
#             config=dict(
#                 model="azure/text-embedding-ada-002", # e.g., "text-embedding-3-small"
#             ),
#         )
#     )
# )


pdf_tool = PDFSearchTool(
    pdf="Ai.pdf",
    config=dict(
        llm=dict(
            provider="azure_openai",
            config=dict(
                model="azure/gpt-4o-2",     # deployment name
                api_key=os.environ["AZURE_API_KEY"],
                base_url=os.environ["AZURE_API_BASE"],
                api_version=os.environ["AZURE_API_VERSION"],
            ),
        ),
        embedder=dict(
            provider="azure_openai",
            config=dict(
                model="azure/text-embedding-ada-002",    # embedding deployment name
                api_key=os.environ["AZURE_API_KEY"],
                base_url=os.environ["AZURE_API_BASE"],
                api_version=os.environ["AZURE_API_VERSION"],
            ),
        ),
    )
)




# os.environ['CURL_CA_BUNDLE'] = ""
# ssl._create_default_https_context = ssl._create_unverified_context


researcher_2 = Agent(
    role= "AI Researcher",
    goal="Gather the latest AI trends from 2024",
    backstory="You are an expert Ai researcher Who writes reports on emerging Ai trends.",
    tools=[pdf_tool],
    verbose=True,
    llm = azure_llm
)
Research_task = Task(
    description= "Search GOOGLE for the top 3 Ai trends in 2024 using online resource.",
    expected_output="a list of of the top 3 trends with a short summary of each",
    agent=researcher_2,
    verbose=True

)

Ai_agent_2 = Crew(
    agents=[researcher_2],
    tasks=[Research_task],
    verbose=True
)


result = Ai_agent_2.kickoff()
