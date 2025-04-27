import json
import re
import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent_router.log')
    ]
)
logger = logging.getLogger(__name__)

# DeepSeek API configuration
DEEPSEEKAI_API_KEY = "sk-75d877a9662b4079a6f7929ee11bbae7"
DEEPSEEKAI_API_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"

class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

class AgentRouter:
    def __init__(self, name: str, capabilities_summary: str, input_schema: Dict[str, Any]):
        self.name = name
        self.capabilities_summary = capabilities_summary
        self.input_schema = input_schema
        self.message_queue = asyncio.Queue()
        logger.info(f"Initialized {self.name} with capabilities: {self.capabilities_summary[:50]}...")
    
    async def evaluate_task(self, content: str) -> Dict[str, Any]:
        """Determine if this agent can handle the requested task"""
        logger.info(f"{self.name} evaluating task: {content[:50]}...")
        
        prompt = f"""
        You are a task evaluator for the {self.name} agent.
        
        The {self.name} agent has the following capabilities:
        {self.capabilities_summary}
        
        User request: {content}
        
        Based solely on the capabilities described, determine if this agent can handle this request.
        Answer with a JSON object: {{"can_handle": true}} or {{"can_handle": false}}
        """
        
        logger.debug(f"{self.name} sending evaluation prompt to LLM")
        response = await self._call_llm(prompt)
        logger.debug(f"{self.name} received evaluation response: {response[:100]}...")
        
        match = re.search(r'\{.*"can_handle"\s*:\s*(true|false).*\}', response, re.DOTALL)
        
        if match:
            try:
                result = json.loads(match.group(0))
                logger.info(f"{self.name} evaluation result: {'CAN handle' if result.get('can_handle', False) else 'CANNOT handle'}")
                if result.get("can_handle", False):
                    return {"request": json.dumps(self.input_schema)}
                return {"request": None}
            except json.JSONDecodeError:
                logger.warning(f"{self.name} failed to parse evaluation response JSON")
                return {"request": None}
        logger.warning(f"{self.name} no valid evaluation response found")
        return {"request": None}

    async def _call_llm(self, prompt: str) -> str:
        """Call to DeepSeek AI API"""
        logger.debug(f"{self.name} calling LLM with prompt length: {len(prompt)}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEKAI_API_KEY}"
        }
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "deepseek-chat",
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        try:
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(DEEPSEEKAI_API_ENDPOINT, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        latency = time.time() - start_time
                        logger.info(f"{self.name} LLM call successful (latency: {latency:.2f}s)")
                        return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"{self.name} Error calling DeepSeek API: {response.status} - {error_text}")
                        return ""
        except Exception as e:
            logger.error(f"{self.name} Exception in LLM call: {str(e)}")
            return ""

class WebSearchAgent(AgentRouter):
    def __init__(self):
        capabilities_summary = """
        This agent can search the web for current information on topics, find factual data,
        retrieve news articles, and gather information from online sources.
        It specializes in information retrieval tasks and fact-finding missions.
        """
        
        input_schema = {
            "query": "[search query]",
            "max_results": 5
        }
        
        super().__init__(
            name="Web Search Agent",
            capabilities_summary=capabilities_summary,
            input_schema=input_schema
        )
    
    async def process(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """Process the search request"""
        logger.info(f"WebSearchAgent processing request: {json.dumps(input_json, indent=2)}")
        
        query = input_json.get("query", "")
        max_results = input_json.get("max_results", 5)
        
        prompt = f"""
        You are a web search simulator. For the query: "{query}", provide {max_results} simulated search results.
        Format your response as a JSON object with the following structure:
        {{
            "results": [
                {{"title": "Result title", "snippet": "Brief description of the result", "url": "https://example.com/page"}},
                ...
            ]
        }}
        """
        
        response = await self._call_llm(prompt)
        logger.debug(f"WebSearchAgent raw response: {response[:200]}...")
        
        # Extract JSON from the response using regex
        match = re.search(r'\{[\s\S]*"results"[\s\S]*\}', response)
        if match:
            try:
                results_json = json.loads(match.group(0))
                logger.info(f"WebSearchAgent returning {len(results_json.get('results', []))} results")
                return results_json
            except json.JSONDecodeError:
                logger.error("WebSearchAgent failed to parse search results JSON")
                return {"error": "Failed to parse search results", "raw_response": response}
        
        logger.error("WebSearchAgent no valid search results found in response")
        return {"error": "No valid search results found", "raw_response": response}

class CodingAgent(AgentRouter):
    def __init__(self):
        capabilities_summary = """
        This agent can write code in various programming languages, debug existing code,
        explain programming concepts, and optimize algorithms. It specializes in software
        development tasks and technical problem-solving.
        """
        
        input_schema = {
            "task": "[coding task description]",
            "language": "[programming language]",
            "constraints": "[any specific requirements]"
        }
        
        super().__init__(
            name="Coding Agent",
            capabilities_summary=capabilities_summary,
            input_schema=input_schema
        )
    
    async def process(self, input_json: Dict[str, Any]) -> Dict[str, Any]:
        """Process the coding request"""
        logger.info(f"CodingAgent processing request: {json.dumps(input_json, indent=2)}")
        
        task = input_json.get("task", "")
        language = input_json.get("language", "python")
        constraints = input_json.get("constraints", "")
        
        prompt = f"""
        You are a coding assistant. Write code for the following task:
        
        Task: {task}
        Language: {language}
        Constraints: {constraints}
        
        Format your response as a JSON object with the following structure:
        {{
            "code": "Your code here with proper escaping for JSON",
            "explanation": "Brief explanation of how the code works"
        }}
        """
        
        response = await self._call_llm(prompt)
        logger.debug(f"CodingAgent raw response: {response[:200]}...")
        
        # Extract JSON from the response using regex
        match = re.search(r'\{[\s\S]*"code"[\s\S]*"explanation"[\s\S]*\}', response)
        if match:
            try:
                code_json = json.loads(match.group(0))
                logger.info("CodingAgent successfully generated code solution")
                return code_json
            except json.JSONDecodeError:
                logger.error("CodingAgent failed to parse code results JSON")
                return {"error": "Failed to parse coding results", "raw_response": response}
        
        logger.error("CodingAgent no valid code found in response")
        return {"error": "No valid code found", "raw_response": response}
class CentralRouter:
    def __init__(self):
        self.agents = {}
        self.message_queue = asyncio.Queue()
        logger.info("CentralRouter initialized")
    
    def register_agent(self, agent_name: str, agent: AgentRouter):
        """Register an agent with the central router"""
        self.agents[agent_name] = agent
        logger.info(f"Registered agent: {agent_name}")
    
    async def broadcast(self, content: str, sender: Optional[str] = None) -> Dict[str, Any]:
        """Broadcast a message to all agents except the sender"""
        logger.info(f"Broadcasting message from {sender or 'user'}: {content[:50]}...")
        results = {}
        
        # First handshake: ask agents if they can handle the task
        agent_responses = {}
        logger.info(f"Starting first handshake with {len(self.agents)} agents")
        
        for agent_name, agent in self.agents.items():
            if agent_name != sender:
                logger.debug(f"Evaluating task with {agent_name}")
                response = await agent.evaluate_task(content)
                agent_responses[agent_name] = response
                logger.debug(f"{agent_name} response: {response}")
        
        # Second handshake: dispatch to willing agents using their requested format
        logger.info("Starting second handshake with willing agents")
        for agent_name, response in agent_responses.items():
            if response["request"] is not None:
                logger.info(f"Agent {agent_name} can handle the task")
                
                try:
                    # Parse the agent's required input format
                    input_schema = json.loads(response["request"])
                    logger.debug(f"{agent_name} input schema: {input_schema}")
                    
                    # Create a formatted input following the agent's schema
                    # We don't hardcode the structure beforehand - we follow what the agent tells us
                    formatted_input = {}
                    
                    # Look through the schema and replace placeholder values 
                    for key, value in input_schema.items():
                        # If we find a placeholder in brackets, replace it with content
                        if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                            # Special case for search query
                            if "query" in key.lower() or "search" in value.lower():
                                formatted_input[key] = content
                            # Special case for task description
                            elif "task" in key.lower() or "description" in value.lower():
                                formatted_input[key] = content
                            # Default case for user content
                            elif "content" in value.lower() or "user" in value.lower():
                                formatted_input[key] = content
                            # Keep the placeholder if we don't know how to replace it
                            else:
                                formatted_input[key] = value
                        else:
                            # Keep non-placeholder values as-is
                            formatted_input[key] = value
                    
                    logger.info(f"Sending formatted input to {agent_name}: {formatted_input}")
                    
                    # Third handshake: send formatted task to agent
                    agent_result = await self.agents[agent_name].process(formatted_input)
                    results[agent_name] = agent_result
                    logger.info(f"Received result from {agent_name}")
                    
                except (json.JSONDecodeError, KeyError) as e:
                    error_msg = f"Failed to format input for {agent_name}: {str(e)}"
                    logger.error(error_msg)
                    results[agent_name] = {"error": error_msg}
            else:
                logger.info(f"Agent {agent_name} cannot handle the task")
        
        logger.info(f"Broadcast completed with {len(results)} results")
        return results

    async def handle_user_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and route to appropriate agents"""
        logger.info(f"Handling user input: {user_input[:50]}...")
        
        # Initial JSON format to broadcast
        initial_json = {"Content": user_input}
        
        # Put in message queue
        await self.message_queue.put((None, json.dumps(initial_json)))
        logger.debug("Added user input to message queue")
        
        all_results = {}
        
        # Process message queue until empty
        while not self.message_queue.empty():
            sender, message = await self.message_queue.get()
            logger.debug(f"Processing message from {sender or 'user'}")
            
            try:
                if isinstance(message, str):
                    message_content = json.loads(message).get("Content", "")
                else:
                    message_content = message.get("Content", "")
            except (json.JSONDecodeError, AttributeError):
                message_content = str(message)
                logger.warning("Failed to parse message content, using string representation")
            
            logger.info(f"Broadcasting message content: {message_content[:50]}...")
            results = await self.broadcast(message_content, sender)
            all_results.update(results)
            
            self.message_queue.task_done()
            logger.debug("Message queue task done")
        
        logger.info(f"Completed processing with {len(all_results)} total results")
        return all_results

async def main():
    # Initialize router and agents
    logger.info("Starting agent router system")
    central_router = CentralRouter()
    web_search_agent = WebSearchAgent()
    coding_agent = CodingAgent()
    
    # Register agents with router
    central_router.register_agent("Web Search Agent", web_search_agent)
    central_router.register_agent("Coding Agent", coding_agent)
    
    # Example user input
    user_input = "Write a Python function to calculate Fibonacci numbers and explain how the Fibonacci sequence appears in nature"
    
    print(f"\nProcessing user input: {user_input}")
    logger.info(f"Main processing user input: {user_input}")
    start_time = time.time()
    results = await central_router.handle_user_input(user_input)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"\nProcessing completed in {processing_time:.2f} seconds")
    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    
    print("\nResults:")
    print(json.dumps(results, indent=2))
    
    # Log performance metrics
    logger.info("Performance Metrics:")
    logger.info(f"- Total processing time: {processing_time:.2f}s")
    if results:
        for agent_name, result in results.items():
            if "error" not in result:
                if agent_name == "Web Search Agent":
                    logger.info(f"- {agent_name} returned {len(result.get('results', []))} results")
                elif agent_name == "Coding Agent":
                    code_length = len(result.get('code', ''))
                    logger.info(f"- {agent_name} returned code solution ({code_length} chars) with explanation")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise
