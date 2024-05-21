from crewai import Agent, Task, Crew, Process
from langchain.llms import Ollama
from dotenv import load_dotenv
import os

ollama_llama3 = Ollama(model="llama3")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

researcher = Agent(
    role='Researcher',
    goal='Find the top 5 trending crypto news articles',
    backstory='You are a crypto research assistant',
    verbose=True,
    allow_delegation=False,
    llm=ollama_llama3  
)

writer = Agent(
    role='Writer',
    goal='Write succinct summaries of the top 5 trending crypto news articles',
    backstory='You are an expert crypto journalist',
    verbose=True,
    allow_delegation=False,
    llm=ollama_llama3
)

task1 = Task(
    description='Investigate the top trending crypto news articles', 
    agent=researcher,
    expected_output='A list of the top 5 trending crypto news articles with links and brief descriptions.'
)

task2 = Task(
    description='Summarise the top trending crypto news articles', 
    agent=writer,
    expected_output='Succinct summaries of the top 5 trending crypto news articles.'
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,
    process=Process.sequential
)

result = crew.kickoff()
print(result)

