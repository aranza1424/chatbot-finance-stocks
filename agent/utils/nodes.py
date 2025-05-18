""" Module to define nodes for the agent"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from agent.utils.tools import tools
from agent.utils.states import MessagesState

load_dotenv(override=True)
os.environ.get("OPENAI_API_KEY")



class Nodes:

   def __init__(self, tools):
      self.llm = ChatOpenAI(model="gpt-4o-mini")
      self.llm_with_tools = self.llm.bind_tools(tools)
      self.sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

   def assistant(self, state: MessagesState):
      return {"messages": [self.llm_with_tools.invoke([self.sys_msg] + state["messages"])]}
   

nodes = Nodes(tools)

