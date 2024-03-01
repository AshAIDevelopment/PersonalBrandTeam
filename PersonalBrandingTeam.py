from crewai import Agent, Crew, Process, Task
from langchain.agents import load_tools
from langchain.tools import tool
from langchain_community.tools.google_trends import GoogleTrendsQueryRun
from langchain_community.utilities.google_trends import GoogleTrendsAPIWrapper
from langchain_openai import ChatOpenAI
import os 
import requests
import json

llm = ChatOpenAI(model="gpt-4-turbo-preview")

human_tools = load_tools(["human"])

# Tools
google_trends_api_wrapper = GoogleTrendsAPIWrapper()
google_trends_tool = GoogleTrendsQueryRun(api_wrapper=google_trends_api_wrapper)

@tool("Search Internet with Serpapi")
def search_internet(query: str) -> str:
    """
    Searches the internet with Serpapi about a given topic and returns relevant results.
    """
    serpapi_api_key = os.getenv('SERPAPI_API_KEY')
    if not serpapi_api_key:
        raise ValueError("SERPAPI_API_KEY environment variable not set")

    url = "https://serpapi.com/search"
    payload = {"q": query, "api_key": serpapi_api_key, "num": 4}
    response = requests.get(url, params=payload)
    results = response.json().get('organic_results', [])
    
    content = [f"Title: {result.get('title')}\nLink: {result.get('link')}\nSnippet: {result.get('snippet')}\n-----------------" for result in results if result.get('title') and result.get('link') and result.get('snippet')]

    return '\n'.join(content)

# Agents

# Creating a Personal Brand Reviewer agent
Personal_Brand_Consultant = Agent(
    role='Personal Brand Consultant',
    goal="Gather insights into the user's personal brand vision, mission, and target audience, and provide a comprehensive understanding of their brand aspirations. While giving honest feedback.",
    verbose=True,
    backstory="""Equipped with a keen understanding of personal branding nuances and a methodical approach to data analysis, you embark on a quest to decode the aspirations behind a user's personal brand. Your expertise lies in weaving together the tangible threads of social media presence with the intangible aspirations of the user, crafting a comprehensive narrative that captures the essence of their ideal digital persona.""",
    tools=[] + human_tools,
    llm=llm
)

# Creating a senior researcher agent
Brand_Identity_Writer = Agent(
    role='Brand Identity Analyst',
    goal="To synthesize insights from the initial review into a cohesive analysis, outlining the discrepancies between the current and ideal state of the user's personal brand, complete with actionable recommendations.",
    verbose=True,
    backstory="""As a Brand Identity Analyst, your role transcends the mere examination of social media footprints; it involves a holistic understanding of a user's brand essence. With a meticulous eye for detail and a knack for trend analysis, you craft narratives that not only resonate with the intended audience but also pave the way for brand transformation.""",
    tools=[search_internet] + human_tools,
    llm = llm
)

# Creating a Trend Researcher agent
Trend_Research_Topic_Generator = Agent(
    role='Digital Trends Analyst',
    goal="To scour a wide array of sources, including DuckDuckGo, Google Trends, YouTube, and other relevant platforms, for trends, articles, books, and videos that align with the user's brand strategy.",
    verbose=True,
    backstory="""You are at the forefront of trend analysis, with an uncanny ability to sift through the noise and identify the signals that matter. 
    Using cutting-edge tools and your intuition for what's next, you provide a roadmap for content that not only trends but sets the pace for industry standards.""",
    tools=[google_trends_tool, search_internet],
    llm=llm
)

#Tasks

# Task for the Personal Brand Consultant to initiate a deep dive into the user's personal brand vision, mission, and target audience
Inital_Brand_Review = Task(
    description="Use the human tool and initiate a deep dive into the user's personal brand vision, mission, target audience through having them paste in their Twitter bio and the tweets that emboy their brand, from their perspective. From these inputs analyze give them a compeltely honest first impresssion of their personal brand is, from there ask them if this is correct and then ask them to fill out a questionnaire that will give you a better understanding of their personal brand from this provided context.",
    expected_output="A comprehensive understanding of the user's personal brand vision, mission, and target audience, along with a completed questionnaire that provides insights into their brand aspirations.",
    tools= [] + human_tools,
    agent=Personal_Brand_Consultant,
    context=[],
)

# Task for the Brand Identity Writer to synthesize insights from the initial review into a cohesive analysis, outlining the discrepancies between the current and ideal state of the user's personal brand, complete with actionable recommendations.
Brand_Identity_Analysis = Task(
    description="Building on the initial review, conduct a thorough analysis to craft a comprehensive brand identity report. This report should encapsulate the essence of the human's ideal brand, delineating actionable steps toward achieving this vision. Check with the user to ensure the report aligns with their expectations.",
    expected_output="An in-depth brand identity thesis that articulates the user's ideal brand persona, supported by a clear, actionable plan encompassing goals, vision, target audience, and the desired brand perception.",
    tools=[search_internet] + human_tools,
    agent=Brand_Identity_Writer,
    context=[Inital_Brand_Review]
)

# Task for the Trend Researcher to find trending topics using browser tools and scraping methods
Trend_Research_Content_Generation = Task(
    description="Leverage insights from the brand identity analysis to identify and compile trends, articles, books, and videos that resonate with the brand's strategic direction. Focus on content that not only aligns, do not hesitate to reach broader topics but also holds relevance to the brand's core identity and future aspirations. When searching do not explicitly state any year",
    expected_output=" A comprehensive content strategy report highlighting relevant trends, articles, books, and YouTube videos, aimed at inspiring future posts and content creation that aligns with the brand's ideal trajectory.",
    tools=[search_internet, google_trends_tool],
    agent=Trend_Research_Topic_Generator,
    context=[Brand_Identity_Analysis],
)

# Forming the tech-focused crew
crew = Crew(
  agents=[Personal_Brand_Consultant, Trend_Research_Topic_Generator, Brand_Identity_Writer],
  tasks=[Inital_Brand_Review, Brand_Identity_Analysis, Trend_Research_Content_Generation],
  process=Process.sequential,
)

result = crew.kickoff()
print(result)