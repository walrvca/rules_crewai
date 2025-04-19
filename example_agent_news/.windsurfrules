## Complete Guide to Creating Agents, Teams, and Multi-Agent Systems with CrewAI

### 1. Key Principles

**Prioritize** the creation of autonomous agents, capable of making decisions and executing tasks independently. **Define** distinct roles for each agent, imitating a team of specialists. **Establish** clear goals for each agent, guiding their decision-making process. **Encourage** collaboration among agents, allowing for task delegation and communication to solve complex problems. **Adopt** CrewAI's flexibility, which allows integration with various tools and LLMs. **Structure** your projects in a modular way to facilitate maintenance and scalability.

### 2. Design Guidelines

**Carefully plan** the roles of the agents, their goals, and the tasks they will perform. **Create** well-defined tasks with clear descriptions and expected outcomes. **Organize** agents and tasks into cohesive teams (Crews) to achieve common objectives. **Choose** the appropriate execution process for your Crew (sequential or hierarchical). **Use** YAML files to configure agents and tasks, promoting clarity and separation of concerns. **Implement** coordination and execution logic in Python files, using CrewAI's classes and decorators. **Start** with simple implementations and move on to more complex systems, using Flows for advanced orchestration.

### 3. Sequential Actions (agents → crews → multi-agents)

**Create** individual agents by defining their `role`, `goal`, and `backstory`. Optionally, **configure** an `llm` for the agent to use.

```python
# Python
from crewai import Agent

# Creating a researcher agent
researcher = Agent(
    role='Web Researcher',
    goal='Find relevant information about a specific topic',
    backstory='You are an experienced researcher with access to various online sources.',
    verbose=True  # To view the agent's process
)
```

**Define** specific tasks (`Task`) and **assign** a responsible agent for each. **Specify** a `description` for the task and the `expected_output`.

```python
# Python
from crewai import Task

# Defining a research task
research_task = Task(
    description='Research the latest trends in artificial intelligence.',
    expected_output='A summary of the top 3 trends with sources.',
    agent=researcher  # Assigns the task to the researcher agent
)
```

**Group** agents and their related tasks into a `Crew`. **Determine** the `process` for executing the tasks within the Crew (e.g., `Process.sequential`).

```python
# Python
from crewai import Crew, Process

# Creating a Crew with an agent and a task
research_team = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True  # To view the Crew's process
)

# To start the Crew
result = research_team.kickoff()
print(f"Team result: {result}")
```

For more complex multi-agent systems, **use** `Flows` to orchestrate the execution of multiple Crews or tasks conditionally and sequentially.

### 4. Essential Structures (agents.yaml, tasks.yaml, crew.py, main.py, .env, tools/)

**Organize** your CrewAI project following a clear directory structure.
*   **`.gitignore`**: Specifies files that Git should ignore.
*   **`pyproject.toml`**: Configuration file for build and package management tools (like Poetry or Pip).
*   **`README.md`**: Project documentation.
*   **`.env`**: Stores sensitive environment variables, such as API keys.
*   **`src/your_project/`**: Contains your project’s source code.
    *   **`__init__.py`**: Makes the directory a Python package.
    *   **`main.py`**: Entry point to run your CrewAI application.
    *   **`crew.py`** (or another relevant name): Defines your Crew’s structure, importing agents and tasks.
    *   **`tools/`**: Contains custom tools (`custom_tool.py`) your agents can use.
    *   **`config/`**: Stores YAML configuration files.
        *   **`agents.yaml`**: Defines agent configurations (role, goal, backstory, llm, tools).
        *   **`tasks.yaml`**: Defines task descriptions and assignments.

### 5. Code Examples by Phase (YAML + Python/CrewBase)

**Define** agents in `config/agents.yaml`:

```yaml
# config/agents.yaml
researcher:
  role: "Web Researcher"
  goal: "Find relevant information about artificial intelligence"
  backstory: "You are an experienced researcher with a focus on emerging technologies."
  llm: "openai/gpt-4o-mini"
  tools:
    - serper_dev_tool
```

**Define** tasks in `config/tasks.yaml`:

```yaml
# config/tasks.yaml
research_task:
  description: "Research the latest trends in artificial intelligence and list the top 3."
  expected_output: "A list of the top 3 trends in AI with a brief description of each."
  agent: "researcher"
```

**Create** the Crew in `src/your_project/crew.py` using `@CrewBase`:

```python
# src/your_project/crew.py
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["SERPER_API_KEY"] = os.environ.get("SERPER_API_KEY")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

@CrewBase
class AIPesquisaCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[SerperDevTool()],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'], agent=self.researcher())

    @crew
    def pesquisa_ia(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )

if __name__ == "__main__":
    pesquisa_crew = AIPesquisaCrew()
    resultado = pesquisa_crew.pesquisa_ia().kickoff()
    print(f"Research result: {resultado}")
```

### 6. Scalable Architecture

**For small projects**, a single Crew with a few agents and well-defined tasks may be sufficient. **Maintain** configuration in YAML files and logic in Python files for organization.

**For large projects**, **divide** functionalities into multiple Crews, each responsible for a specific aspect of the system. **Use** `Flows` to orchestrate the interaction between these Crews and to define complex workflows with conditional logic. **Create** custom tools in the `tools/` directory to meet the specific needs of your project. **Consider** using a manager agent (`manager_agent`) in hierarchical processes to coordinate larger teams.

### 7. YAML-Python Integration Pattern via @CrewBase, @agent, @task, @flow

As demonstrated in section 5, `@CrewBase` allows defining Crews using YAML files to configure agents and tasks. The `@agent` and `@task` decorators within a `@CrewBase` class load the corresponding YAML configurations, creating instances of `Agent` and `Task` automatically.

For `Flows`, use `@flow` to define a class representing a workflow. Use `@start` to mark the starting point of the flow and `@listen` to define subsequent steps, indicating which step the current one depends on.

```python
# Conceptual example of Flow
from crewai.flow import Flow, start, listen
from pydantic import BaseModel

class FlowState(BaseModel):
    data: dict = {}
    result: str = ""

@Flow(state=FlowState)
class MyFlow:
    @start()
    def initiate(self):
        self.state.data = {"parameter": "initial value"}
        return "Flow initiated"

    @listen(initiate)
    def process_data(self, output_initiate):
        self.state.result = f"Data processed with: {self.state.data['parameter']} and initiate output: {output_initiate}"
        return self.state.result

if __name__ == "__main__":
    flow = MyFlow()
    final_result = flow.kickoff()
    print(f"Flow result: {final_result}")
```

### 8. LLM and Tools Configuration (demonstrate code as per item 7)

In the `config/agents.yaml` file, **specify** the desired `llm` for each agent:

```yaml
# config/agents.yaml
writer:
  role: "Content Writer"
  goal: "Create engaging and informative content"
  backstory: "You are a talented writer with experience in various formats."
  llm: "openai/gpt-4"
```

To use tools, **list** them in the `tools` section of the agent in `agents.yaml`. The tools must be available and configured in your Python environment.

```yaml
# config/agents.yaml
analyst:
  role: "Data Analyst"
  goal: "Analyze data and provide actionable insights"
  backstory: "You are a data analyst with expertise in extracting knowledge from large volumes of information."
  llm: "openai/gpt-4o"
  tools:
    - serper_dev_tool
    - spreadsheet_tool  # Example of a custom tool
```

In your Python code (`crew.py`), **import** and initialize the tools. When using `@CrewBase`, the tools listed in the YAML will be automatically associated with the agents.

```python
# src/your_project/crew.py
from crewai import Agent
from crewai.project import CrewBase, agent
from rag_tool import RagTool

class DocumentacaoCrew(CrewBase):
    agents_config = 'config/agents.yaml'
    # tasks_config ...

    @agent
    def analyst(self) -> Agent:
        rag_tool = RagTool(documentos_path="./documentos.txt")
        return Agent(
            config=self.agents_config['analyst'],
            tools=[rag_tool],
            verbose=True
        )

    # ...
```

### 9. Memory (Short-Term, Long-Term, Entity, Contextual) and RAG (demonstrate code as per item 7)

Memory allows agents to retain information over time. **Enable** short-term memory by defining `memory=True` when creating an `Agent` or `Crew`.

```yaml
# config/agents.yaml
editor:
  role: "Technical Editor"
  goal: "Write clear and precise documentation"
  backstory: "You are an experienced technical editor with attention to detail."
  llm: "openai/gpt-3.5-turbo"
  memory: True
```

For long-term memory and Retrieval-Augmented Generation (RAG) capabilities, **integrate** libraries like LangChain and LlamaIndex. **Create** custom tools that use these libraries to search and incorporate relevant information into the context of the agents' tasks.

```python
# src/your_project/tools/rag_tool.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from crewai.tools import BaseTool

class RagTool(BaseTool):
    name = "rag_tool"
    description = "Useful for searching relevant information from documents."

    def __init__(self, documentos_path: str):
        super().__init__()
        self.loader = TextLoader(documentos_path)
        self.documents = self.loader.load()
        self.embeddings = OpenAIEmbeddings()
        self.db = Chroma.from_documents(self.documents, self.embeddings)
        self.retriever = self.db.as_retriever()
        self.qa = RetrievalQA.from_chain_type(llm=None, chain_type="stuff", retriever=self.retriever)

    def _run(self, query: str) -> str:
        return self.qa.run(query)

# src/your_project/crew.py
from crewai import Agent
from crewai.project import CrewBase, agent
from rag_tool import RagTool

class ConteudoCrewRAG(CrewBase):
    agents_config = 'config/agents.yaml'
    # tasks_config ...

    @agent
    def writer_rag(self) -> Agent:
        rag_tool = RagTool(documentos_path="./documentos_extensos.txt")
        return Agent(
            config=self.agents_config['writer_rag'],
            tools=[rag_tool],
            verbose=True
        )

    # ...
```

### 10. AI-Friendly Coding Practices

**Use** descriptive names for variables, functions, classes, and files to facilitate code understanding. **Add** type hints to improve readability and help in error detection. **Include** detailed comments explaining the logic and purpose of complex code sections. **Provide** rich context in error messages and logs to facilitate debugging and system monitoring. **Structure** your code in a modular and organized way.

### 11. Agent Communication and Collaboration (demonstrate code as per item 7)

Communication and collaboration among agents occur mainly through the definition of sequential or hierarchical tasks within a Crew, where the output of one task serves as context for the next. Task delegation is controlled by assigning tasks to specific agents. An agent's ability to delegate tasks to others can be enabled with `allow_delegation=True` in the agent's configuration.

```python
# src/your_project/crew.py
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew

class ConteudoCrewComOuvinte(CrewBase):
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(config=self.agents_config['researcher'], verbose=True, allow_delegation=True)

    @agent
    def writer(self) -> Agent:
        return Agent(config=self.agents_config['writer'], verbose=True)

    @task
    def research(self) -> Task:
        return Task(config=self.tasks_config['research'], agent=self.researcher())

    @task
    def writing(self) -> Task:
        return Task(config=self.tasks_config['writing'], agent=self.writer(), context=[self.research()])

    @crew
    def content_creation(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
```

### 12. Complex Crew Examples (demonstrate code as per item 7)

**Create** Crews with multiple specialized agents working together to solve complex tasks. **Define** a clear workflow, using `Process.sequential` for interdependent tasks or `Process.hierarchical` with a manager agent to coordinate a group of agents.

```python
# src/your_project/crew.py
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew

class FinancialAnalysisCrew(CrewBase):
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def market_analyst(self) -> Agent:
        return Agent(config=self.agents_config['market_analyst'], tools=[SerperDevTool()], verbose=True)

    @agent
    def risk_analyst(self) -> Agent:
        return Agent(config=self.agents_config['risk_analyst'], verbose=True)

    @agent
    def reporter(self) -> Agent:
        return Agent(config=self.agents_config['reporter'], verbose=True)

    @task
    def analyze_trends(self) -> Task:
        return Task(config=self.tasks_config['analyze_trends'], agent=self.market_analyst())

    @task
    def evaluate_risks(self) -> Task:
        return Task(config=self.tasks_config['evaluate_risks'], agent=self.risk_analyst(), context=[self.analyze_trends()])

    @task
    def generate_report(self) -> Task:
        return Task(config=self.tasks_config['generate_report'], agent=self.reporter(), context=[self.evaluate_risks()])

    @crew
    def complete_analysis(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )
```

### 13. Versioning & Dependencies

**Adopt** Semantic Versioning (SemVer) to version your project (MAJOR.MINOR.PATCH). **Register** all dependencies of your project (including `crewai` and `crewai-tools`) in the `requirements.txt` file.

```text
# requirements.txt
crewai
crewai-tools
langchain  # If using
llama-index # If using
python-dotenv
```

**Use** the command `pip freeze > requirements.txt` to generate this file automatically after installing all your dependencies.

### 14. Orchestration & Flows

**Use** CrewAI Flows to create complex workflows that involve multiple steps, possibly with conditional logic and loops. **Define** the flow state using `pydantic.BaseModel`. **Use** the `@start` and `@listen` decorators to define the execution sequence of the flow steps. **Employ** the `@router` decorator to implement conditional logic, directing the flow based on the state or output of a previous step.

```python
# src/your_project/flows/analysis_flow.py
from crewai.flow import Flow, start, listen, router
from crewai import Crew, Agent, Task, Process
from pydantic import BaseModel

class AnalysisState(BaseModel):
    initial_report: str = ""
    approved: bool = False
    final_report: str = ""

@Flow(state=AnalysisState)
class CompleteAnalysisFlow:
    @start()
    def generate_initial_report(self):
        # ... logic to generate the initial report using a Crew ...
        self.state.initial_report = "Initial report generated."
        return self.state.initial_report

    @listen(generate_initial_report)
    def verify_quality(self, report):
        # ... logic to verify the report quality ...
        if "quality ok" in report:
            self.state.approved = True
            return "approved"
        else:
            self.state.approved = False
            return "revise"

    @listen(verify_quality, trigger="revise")
    def request_revision(self, _):
        # ... logic to request revision using another Crew ...
        return "Revision requested."

    @listen(verify_quality, trigger="approved")
    def generate_final_report(self, report):
        self.state.final_report = f"Final report: {report}"
        return self.state.final_report

# In main.py:
# from flows.analysis_flow import CompleteAnalysisFlow
# flow = CompleteAnalysisFlow()
# result = flow.kickoff()
# print(result)
```

### 15. Event Listeners

**Implement** the `BaseEventListener` class to create custom listeners that react to specific events during the execution of your Crew or Flow. This allows registering logs, collecting metrics, or triggering external actions in response to events like the start or completion of a task.

```python
# src/your_project/listeners.py
from crewai.event_listeners import BaseEventListener
from crewai.events import TaskCompletedEvent

class MyListener(BaseEventListener):
    def on_task_completed(self, event: TaskCompletedEvent):
        print(f"Task completed: {event.task.description}")
        # Add logic here for logs, metrics, or external triggers

# src/your_project/crew.py
from crewai import Crew, Process
from listeners import MyListener

class MyCrewWithListener(Crew):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, event_listeners=[MyListener()])

# Or in a Flow:
from crewai.flow import Flow
from listeners import MyListener

class MyFlowWithListener(Flow):
    event_listeners = [MyListener()]
    # ... flow definition ...
```

### 16. Limits & Rate Limiting

**Define** the `max_rpm` parameter (maximum requests per minute) in `Agent` or `Crew` objects to avoid hitting API rate limits. **Implement** fallback mechanisms, such as retries with exponential backoff, to handle rate limit errors (status code 429).

```python
# Python
from crewai import Agent

researcher = Agent(
    role='Web Researcher',
    goal='Find relevant information about a specific topic',
    backstory='You are an experienced researcher with access to various online sources.',
    verbose=True,
    max_rpm=10  # Limits to 10 requests per minute
)

from crewai import Crew, Process
import time

class ResilientTeam(Crew):
    def kickoff(self, inputs=None, max_retries=3, initial_backoff=1):
        retries = 0
        while retries < max_retries:
            try:
                return super().kickoff(inputs=inputs)
            except Exception as e:
                if "429" in str(e):
                    retries += 1
                    backoff_time = initial_backoff * (2 ** (retries - 1))
                    print(f"Rate limit hit. Trying again in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                else:
                    raise  # Relevant other errors
        print("Maximum number of attempts reached after rate limit errors.")
        return None

research_team = ResilientTeam(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True
)
```

### 17. Performance & RAG

**Perform** profiling of LLM calls to identify performance bottlenecks. **Optimize** the embedding creation process and the chunk size of documents for RAG (a common size ranges between 256 and 512 tokens, but this depends on your use case and LLM). **Adjust** the `similarity_top_k` parameter to control the number of documents retrieved for each query. **Integrate** RAG with an efficient storage solution for embeddings (e.g., a vector database like Chroma, Pinecone, or Weaviate).

```python
# src/your_project/tools/rag_tool.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from crewai.tools import BaseTool

class OptimizedRagTool(BaseTool):
    name = "optimized_rag_tool"
    description = "Useful for searching relevant information from documents with optimized embeddings."

    def __init__(self, documentos_path: str):
        super().__init__()
        self.loader = TextLoader(documentos_path)
        self.documents = self.loader.load()
        self.embeddings = OpenAIEmbeddings()
        self.db = Chroma.from_documents(self.documents, self.embeddings, chunk_size=512) # Optimized chunk size
        self.retriever = self.db.as_retriever(search_kwargs={"k": 5}) # Adjusted similarity_top_k
        self.qa = RetrievalQA.from_chain_type(llm=None, chain_type="stuff", retriever=self.retriever)

    def _run(self, query: str) -> str:
        return self.qa.run(query)

# src/your_project/crew.py
from crewai import Agent
from crewai.project import CrewBase, agent
from rag_tool import OptimizedRagTool

class ConteudoCrewRAG(CrewBase):
    agents_config = 'config/agents.yaml'
    # tasks_config ...

    @agent
    def writer_rag(self) -> Agent:
        rag_tool = OptimizedRagTool(documentos_path="./documentos_extensos.txt")
        return Agent(
            config=self.agents_config['writer_rag'],
            tools=[rag_tool],
            verbose=True
        )

    # ...