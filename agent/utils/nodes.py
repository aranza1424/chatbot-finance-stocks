""" Module to define nodes for the agent"""
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, END
from langgraph.types import Send
from langchain_core.messages import AIMessage
import chromadb

from agent.utils.tools import tools_main
from agent.utils.states import *

#Load env variables
load_dotenv(override=True)

#read YAMl
with open('prompts.yml', 'r') as f:
    data_prompt = yaml.safe_load(f)
with open('config_parameters.yml', 'r') as f:
    data_config = yaml.safe_load(f)


#Define YAML constants
MODEL_LLM = data_config["LLM"]["MODEL_LLM"]
VECTOR_DB_PATH = data_config["chroma"]["VECTOR_DB_PATH"]
MODEL_EMBEDDINGS = data_config["embeddings"]["MODEL_EMBEDDINGS"]
COLLECTION_NAME = data_config["chroma"]["COLLECTION_NAME"]


#Define YAML prompts
PROMPT_SYSTEM_MAIN = data_prompt["prompt_agent_main"]["PROMPT_SYSTEM_MAIN"]
PROMPT_MULTIPLE_QUESTIONS = data_prompt["prompt_agent_query"]["PROMPT_MULTIPLE_QUESTIONS"]
PROMPT_GET_QUERY_ANSWER = data_prompt["prompt_agent_query"]["PROMPT_GET_QUERY_ANSWER"]




"""
------------------------------------------
NODES for main Agent
"""

class MainNodes:

   def __init__(self):
      self.llm = ChatOpenAI(model=MODEL_LLM, temperature=0)

   
   def decide_tool(self, state: OverallQueryState):
      user_input = state["messages"][0].content
      response = self.llm.bind_tools(tools_main, 
                                     parallel_tool_calls=False).invoke([SystemMessage(content=PROMPT_SYSTEM_MAIN)]+state["messages"])
      return {"messages": [response], "question": user_input}
   
   def route_message(self, state: OverallQueryState) -> Literal[END, "execute_tools"]:
      """Check message to call tool"""
      message = state['messages'][-1]
      if not message.tool_calls:
         return END
      
      tool_call = message.tool_calls[0]
      if tool_call['name'] == "query":
         return "execute_tools"
      else:
         print(message)
         raise ValueError(f"Unknown tool call: {tool_call['name']}")
      
main_nodes = MainNodes()

"""
------------------------------------------
"""


"""
------------------------------------------
NODES for Query Agent
"""

class QueryNodes:

   def __init__(self):
      self.llm = ChatOpenAI(model=MODEL_LLM, temperature=0)
      self.embeddings = OpenAIEmbeddings(model=MODEL_EMBEDDINGS)
      self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
      self.chroma_collection = self.chroma_client.get_collection(name=COLLECTION_NAME)

   def generate_questions(self, state: OverallQueryState):
      prompt = PROMPT_MULTIPLE_QUESTIONS.format(question=state["question"])
      response = self.llm.with_structured_output(Context).invoke(prompt)
      return {"context": response.context}
   
   def continue_to_query(self, state: OverallQueryState):
      return [Send("generate_query", {"subquestion": s}) for s in state["context"]]
   
   def generate_query(self, state: QueryState):
      single_vector = self.embeddings.embed_query(state["subquestion"])
      query_results = self.chroma_collection.query(query_embeddings=single_vector,n_results=2)
      result_query = query_results["documents"][0]

      return {"answers": result_query}
   
   def get_final_query_response(self, state: OverallQueryState):
      get_unique_answers = set(state["answers"])
      answers = "\n\n".join(get_unique_answers)
      
      prompt = PROMPT_GET_QUERY_ANSWER.format(question=state["question"], companies=answers)
      response = self.llm.with_structured_output(FinalAnswerQuery).invoke(prompt)
      return {"messages": [AIMessage(content=response.answer)]}

query_nodes = QueryNodes()

"""
------------------------------------------
"""
