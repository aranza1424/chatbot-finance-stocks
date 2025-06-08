from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.agent_graph import AgentGraph

#create graph
main_agent = AgentGraph()

#define question schema
class QuestionRequest(BaseModel):
    question: str
    thread_id: str

# Definimos la estructura de datos para las respuestas
class AnswerResponse(BaseModel):
    answer: str

app = FastAPI(title="Finance Stock Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.post("/API/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    
    response_agent = await main_agent.get_async_response(request.question, request.thread_id)
    
    return AnswerResponse(
        answer=response_agent
    )
