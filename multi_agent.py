from ctypes import _NamedFuncPointer
import os
import asyncio
import subprocess
from chainlit import user_session

from dotenv import load_dotenv
from autogen import GroupChat, GroupChatManager, AssistantAgent, UserProxyAgent
from langchain.prompts import ChatPromptTemplate
from autogen.agentchat.groupchat import AuthorRole
from autogen.agentchat.conversable_agent import ConversableAgent
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies.termination.termination_strategy import TerminationStrategy
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.kernel import Kernel

#===Load environment variables ==
load_dotenv()

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


# === Initialize Kernel and attach Azure OpenAI Service====
kernel = kernel()
chat_service = AzureChatCompletion(
    deployment_name=DEPLOYMENT_NAME,
    endpoint=ENDPOINT,
    api_key=API_KEY,
    api_version=API_VERSION
)
kernel.add_service(chat_service)

#=== Termination Startegy =====

class ApprovalTerminationStrategy(TerminationStrategy):
    """A strategy for determining when an agent should terminate."""
 
    async def should_agent_terminate(self, agent, history):
        """Check if the agent should terminate."""
        #Check if user said "APPROVED" in any message
        for message in reversed(history.messages):
            if message.role == AuthorRole.user and "APPROVED" in message.content.lower():
                #Extract HTML from Software Engineer
                if "SoftwareEngineer" in agent.name:
                   for msg in reversed(history.message):
                    if msg.role == AuthorRole.assistant and "<html>" in msg.content.lower():
                        with open("index.html","w") as f:
                            f.write(msg.content)
                            print("[+] HTML extraxted and saved to index.html")

                            #call push script
                            subprocess.run(["bash", "push_to_github.sh"])
                return True
        return False

#Agent Instructions (Persona Prompts)
business_analyst_prompts = """You are a Business Analyst which will take the requirements from the user (also known as a 'customer') and create a project plan for creating the requested app. The Business Analyst understands the user requirements and creates detailed documents with requirements and costing. The documents should be usable by the SoftwareEngineer as a reference for implementing the required features, and by the Product Owner for reference to determine if the application delivered by the Software Engineer meets all of the user's requirements."""
software_engineer_prompts = """You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear."""
product_owner_prompts ="""You are a Software Engineer, and your goal is create a web app using HTML and JavaScript by taking into consideration all the requirements given by the Business Analyst. The application should implement all the requested features. Deliver the code to the Product Owner for review when completed. You can also ask questions of the BusinessAnalyst to clarify any requirements that are unclear."""

#=== Create ChatCompletionAgents

#Business Analyst Persona
business_analyst = ChatCompletionAgent(
    kernel=kernel,
    prompt_template=ChatPromptTemplate(system_prompt=business_analyst_prompts),
    name="BusinessAnalyst"
)

#Software Engineer Persona
software_engineer = ChatCompletionAgent(

    kernel=kernel,
    prompt_template=ChatPromptTemplate(system_prompt=software_engineer_prompts),
    name="SoftwareEngineer"
)

#Product Owner Persona
product_owner = ChatCompletionAgent(

    kernel=kernel,
    prompt_template=ChatPromptTemplate(system_prompt=product_owner_prompts),
    name="ProductOwner"
)


    # === Group Chat Set Up ===
  #Create the AgentGroupChat with all agents 
   # group_chat = AgentGroupChat(
       # agents=[business_analyst,software_engineer,product_owner],
        #execution_settings=AgentExecutionSettings(
           # terminnation=ApprovalTerminationStrategy()
       # )
    #)

# Agent Group Chat
termination_strategy = ApprovalTerminationStrategy()
group_chat = AgentGroupChat(
    agents=[business_analyst, software_engineer, product_owner],
    execution_settings={"termination_strategy": termination_strategy}
)

# Multi-Agent Execution

async def run_multi_agent(user_input: str):
   group_chat.add_chat_message(AuthorRole.User, user_input)
   print("\n--- Agent conversation Started ---\n")
   
   results = await group_chat.invoke()
   for content in results:
       print(f"# {content.role} - {content.name or '*'}: '{content.content}'")
       
       print("\n--Agent conversation Started --\n")


   if _NamedFuncPointer == "_main_":
      user_session.input = input("Enter your request for the agent group:")
      asyncio.run(run_multi_agent(user_input))
      










