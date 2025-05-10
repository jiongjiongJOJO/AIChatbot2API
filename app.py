import re
import os
import uuid
import time
import json
import httpx
import uvicorn
import hashlib
import secrets
import logging
from pydantic import BaseModel
from dotenv import load_dotenv
from fake_useragent import UserAgent
from fastapi.security import APIKeyHeader
from typing import List, Literal, Optional
from fastapi.responses import StreamingResponse
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Depends

# Load .env file
load_dotenv()

# Retrieve environment variables
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in .env file")

ENABLE_CORS = os.getenv("ENABLE_CORS", "True").lower() in ("true", "1", "yes")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_CHARS = int(os.getenv("MAX_CHARS", "80000"))
RANDOM_UA = os.getenv("RANDOM_UA", "False").lower() in ("true", "1", "yes")

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Suppress httpx and httpcore debug logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS if configured
if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS enabled")
else:
    logger.info("CORS disabled")

# Constants
api_domain = "https://ai-chatbot.top"
default_user_agent = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:136.0) "
    "Gecko/20100101 Firefox/136.0"
)
ua = UserAgent()

# Supported models
supported_models = ["DeepSeek-R1", "DeepSeek-R1-Web"]
model_to_config = {
    "DeepSeek-R1": {"model": "deepseek-huoshan", "isWebSearchEnabled": False},
    "DeepSeek-R1-Web": {"model": "deepseek-huoshan", "isWebSearchEnabled": True},
}

# Device ID and UA mapping
device_ua_map = {}


# Utility functions
def nanoid(size=21):
    url_alphabet = "abcdefgh0ijkl1mno2pqrs3tuv4wxyz5ABCDEFGH6IJKL7MNO8PQRS9TUV-WXYZ_"
    return "".join(secrets.choice(url_alphabet) for _ in range(size))


def generate_device_id():
    return f"{uuid.uuid4().hex}_{nanoid(20)}"


def get_user_agent(device_id: str) -> str:
    if not RANDOM_UA:
        return default_user_agent
    if device_id not in device_ua_map:
        device_ua_map[device_id] = ua.random
    return device_ua_map[device_id]


def generate_sign(chat_id: str, timestamp: int) -> str:
    message = f"{chat_id}{timestamp}@!~chatbot.0868"
    return hashlib.md5(message.encode("utf-8")).hexdigest()


async def get_chat_id() -> str:
    cookies = {
        '_ga_HVMZBNYJML': 'GS1.1.1742013194.1.0.1742013194.0.0.0',
        '_ga': 'GA1.1.1029622546.1742013195',
    }
    headers = {
        "User-Agent": default_user_agent,
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "Referer": "https://ai-chatbot.top/",
        "RSC": "1",
        "Next-Router-State-Tree": '["",{"children":["(chat)",{"children":["__PAGE__",{}, "/", "refresh"]}]},null,"refetch"]',
        "DNT": "1",
        "Sec-GPC": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Priority": "u=0",
    }
    params = {"_rsc": "l4cx"}
    async with httpx.AsyncClient(http2=True) as client:
        response = await client.get(f"{api_domain}/", params=params, cookies=cookies, headers=headers, timeout=30)
        chat_ids = re.findall(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}',
                              response.text)
        return chat_ids[-1]


# API key validation
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(authorization: str = Depends(api_key_header)):
    if not authorization:
        logger.error("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    api_key = authorization.replace("Bearer ", "").strip() if authorization.startswith("Bearer ") else authorization
    if api_key != API_KEY:
        logger.error(f"Invalid API key: {api_key}")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key


# Request models
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    max_tokens: Optional[int] = None


# Response generation
def generate_chunk(_id: str, created: int, model: str, content: str = "", finish_reason: Optional[str] = None):
    chunk = {
        "id": _id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}]
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def truncate_messages(messages: List[Message], max_chars: int = MAX_CHARS) -> List[Message]:
    total_chars = sum(len(msg.content) for msg in messages)
    if total_chars <= max_chars:
        return messages
    truncated = []
    current_chars = 0
    for msg in reversed(messages):
        if current_chars + len(msg.content) <= max_chars:
            truncated.insert(0, msg)
            current_chars += len(msg.content)
        else:
            break
    logger.info(f"Truncated messages: original {total_chars}, truncated {current_chars}")
    return truncated


async def stream_response(request: ChatCompletionRequest, device_id: str, chat_id: str, timestamp: int, sign: str):
    truncated_messages = truncate_messages(request.messages)
    messages = [{"role": msg.role, "content": msg.content} for msg in truncated_messages]
    payload = {
        "id": chat_id,
        "messages": messages,
        "selectedChatModel": model_to_config[request.model]["model"],
        "isDeepThinkingEnabled": True,
        "isWebSearchEnabled": model_to_config[request.model]["isWebSearchEnabled"],
    }
    headers = {
        "User-Agent": get_user_agent(device_id),
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "Referer": f"https://ai-chatbot.top/chat/{chat_id}",
        "Content-Type": "application/json",
        "currentTime": str(timestamp),
        "sign": sign,
        "Origin": "https://ai-chatbot.top",
        "DNT": "1",
        "Sec-GPC": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Priority": "u=0",
    }
    cookies = {
        '_ga_HVMZBNYJML': 'GS1.1.1742013194.1.1.1742013780.0.0.0',
        '_ga': 'GA1.1.1029622546.1742013195',
    }
    _id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    logger.info(f"Starting stream response: chat_id={chat_id}, UA={headers['User-Agent']}")
    yield generate_chunk(_id, created, request.model, "")

    async with httpx.AsyncClient(http2=True, timeout=900) as client:
        async with client.stream("POST", f"{api_domain}/api/chat", json=payload, headers=headers,
                                 cookies=cookies) as response:
            if response.status_code != 200:
                logger.error(f"API error: status_code={response.status_code}")
                yield generate_chunk(_id, created, request.model, f"Error: HTTP {response.status_code}", "stop")
                return
            thinking = False
            content_parts = []
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("g:"):
                    if not thinking:
                        thinking = True
                        content_parts.append("<think>")
                        yield generate_chunk(_id, created, request.model, "<think>")
                    content = line[2:].strip().replace('"', '').replace("\\n", "\n")
                    content_parts.append(content)
                    yield generate_chunk(_id, created, request.model, content)
                elif line.startswith("0:"):
                    if thinking:
                        thinking = False
                        content_parts.append("</think>")
                        yield generate_chunk(_id, created, request.model, "</think>")
                    content = line[2:].strip().replace('"', '').replace("\\n", "\n")
                    content_parts.append(content)
                    yield generate_chunk(_id, created, request.model, content)
            if thinking:
                content_parts.append("</think>")
                yield generate_chunk(_id, created, request.model, "</think>")
            yield generate_chunk(_id, created, request.model, "", "stop")
            logger.info(f"Stream completed: chat_id={chat_id}, content={''.join(content_parts)}")


# not_stream_response
async def not_stream_response(request: ChatCompletionRequest, device_id: str, chat_id: str, timestamp: int, sign: str):
    truncated_messages = truncate_messages(request.messages) if callable(truncate_messages) else truncate_messages(request.messages)
    print(truncated_messages)
    messages = [{"role": msg.role, "content": msg.content} for msg in truncated_messages]
    payload = {
        "id": chat_id,
        "messages": messages,
        "selectedChatModel": model_to_config[request.model]["model"],
        "isDeepThinkingEnabled": True,
        "isWebSearchEnabled": model_to_config[request.model]["isWebSearchEnabled"],
    }
    headers = {
        "User-Agent": get_user_agent(device_id),
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2",
        "Referer": f"https://ai-chatbot.top/chat/{chat_id}",
        "Content-Type": "application/json",
        "currentTime": str(timestamp),
        "sign": sign,
        "Origin": "https://ai-chatbot.top",
        "DNT": "1",
        "Sec-GPC": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "Priority": "u=0",
    }
    cookies = {
        '_ga_HVMZBNYJML': 'GS1.1.1742013194.1.1.1742013780.0.0.0',
        '_ga': 'GA1.1.1029622546.1742013195',
    }
    _id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    logger.info(f"Starting non-stream response: chat_id={chat_id}, UA={headers['User-Agent']}")

    async with httpx.AsyncClient(http2=True, timeout=900) as client:
        response = await client.post(f"{api_domain}/api/chat", json=payload, headers=headers, cookies=cookies)

    if response.status_code != 200:
        logger.error(f"API error: status_code={response.status_code}")
        return {"error": {"message": f"Error: HTTP {response.status_code}", "type": "invalid_request_error"}}
    else:
        contents = ''
        think_content = ''
        print(response.text)
        for line in response.iter_lines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("g:"):
                content = line[3:-1].replace('\\n', '\n')
                think_content += content
                continue
            elif line.startswith("0:"):
                if think_content:
                    contents = '<think>' + think_content + '</think>'
                    think_content = ''
                content = line[3:-1].replace('\\n', '\n')
                contents += content
                continue
            elif line.startswith("e:"):
                continue
            elif line.startswith("d:"):
                continue
            else:
                continue
        print({
            "id": _id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": contents}, "finish_reason": None}],
        })
        return {
            "id": _id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": contents}, "finish_reason": None}],
        }


# Endpoints
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, _: str = Depends(verify_api_key)):
    if request.model not in supported_models:
        request.model = "DeepSeek-R1"
    device_id = generate_device_id()
    chat_id = await get_chat_id()
    timestamp = int(time.time() * 1000)
    sign = generate_sign(chat_id, timestamp)
    logger.info(f"Chat request: model={request.model}, stream={request.stream}, chat_id={chat_id}")

    if request.stream:
        return StreamingResponse(stream_response(request, device_id, chat_id, timestamp, sign),
                                 media_type="text/event-stream")
    else:
        # not_stream_response(request, device_id, chat_id, timestamp, sign)
        return await not_stream_response(request, device_id, chat_id, timestamp, sign)


@app.get("/v1/models")
async def list_models(_: str = Depends(verify_api_key)):
    current_time = int(time.time())
    models = [
        {
            "id": model,
            "object": "model",
            "created": current_time,
            "owned_by": "aichatbot",
        } for model in supported_models
    ]
    logger.info("Returning model list")
    return {"object": "list", "data": models}


@app.get("/health")
async def health_check():
    chat_id = await get_chat_id()
    return {"status": "ok", "session": "active" if chat_id else "inactive"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
