from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os
import yaml

load_dotenv()

# Load agent definitions from YAML
with open(os.path.join(os.path.dirname(__file__), '../config/agents.yaml'), 'r') as f:
    agents_config = yaml.safe_load(f)

# Load task definitions from YAML
with open(os.path.join(os.path.dirname(__file__), '../config/tasks.yaml'), 'r') as f:
    tasks_config = yaml.safe_load(f)

# Instantiate tools
serper_tool = SerperDevTool()

# Create agents
def create_agents():
    agents = {}
    for name, conf in agents_config.items():
        tools = [serper_tool] if 'serper_dev_tool' in conf.get('tools', []) else []
        agents[name] = Agent(
            role=conf['role'],
            goal=conf['goal'],
            backstory=conf['backstory'],
            llm=conf['llm'],
            tools=tools,
            verbose=True
        )
    return agents

# Create tasks
def create_tasks(agents):
    tasks = {}
    for name, conf in tasks_config.items():
        tasks[name] = Task(
            description=conf['description'],
            expected_output=conf['expected_output'],
            agent=agents[conf['agent']],
            verbose=True
        )
    return tasks

class AINewsCrew:
    def __init__(self):
        agents = create_agents()
        tasks = create_tasks(agents)
        self.crew = Crew(
            agents=list(agents.values()),
            tasks=list(tasks.values()),
            process=Process.sequential,
            verbose=True
        )

    def kickoff(self):
        return self.crew.kickoff()
