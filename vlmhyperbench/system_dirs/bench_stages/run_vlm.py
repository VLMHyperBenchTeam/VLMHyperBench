import os
import asyncio
import logging
import json
import time
import pandas as pd
from typing import List, Dict, Any

from task_registry.manager import RegistryManager
from dataset_factory.factory import DatasetFactory
from prompt_manager.manager import PromptManager
from api_wrapper.client import AsyncModelClient
from api_wrapper.schemas.model import ModelLoadRequest
from api_wrapper.schemas.chat import ChatCompletionRequest, ChatMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_vlm")

async def process_item(
    semaphore: asyncio.Semaphore,
    client: AsyncModelClient,
    item: Any,
    prompt_manager: PromptManager,
    run_config: Any,
) -> Dict[str, Any]:
    """
    Processes a single dataset item: resolves prompts, calls the model, and returns the result.
    """
    async with semaphore:
        try:
            # 1. Resolve prompts using PromptManager
            # item.metadata should contain doc_type or other relevant info for PromptManager
            item_metadata = getattr(item, "metadata", {})
            if not isinstance(item_metadata, dict):
                item_metadata = {"doc_type": str(item_metadata)}
            
            # Context for template rendering (e.g. question, options)
            context = item.model_dump() if hasattr(item, "model_dump") else item.__dict__
            
            resolved_prompts = prompt_manager.get_prompts(
                item_metadata=item_metadata,
                context=context
            )
            
            system_prompt = resolved_prompts.get("system_prompt", "")
            user_prompt = resolved_prompts.get("user_prompt", "")
            
            # 2. Prepare ChatCompletionRequest
            messages = []
            if system_prompt:
                messages.append(ChatMessage(role="system", content=system_prompt))
            
            # Handle multimodal content (image + text)
            # Assuming item has image_path and we need to pass it to API Wrapper
            # API Wrapper Data Plane should handle image loading/encoding
            content = [
                {"type": "text", "text": user_prompt}
            ]
            
            if hasattr(item, "image_path") and item.image_path:
                content.append({"type": "image_url", "image_url": {"url": f"file://{item.image_path}"}})
            
            messages.append(ChatMessage(role="user", content=content))
            
            request = ChatCompletionRequest(
                model=run_config.model_name,
                messages=messages,
                temperature=run_config.temperature if hasattr(run_config, "temperature") else 0.0,
                max_tokens=run_config.max_tokens if hasattr(run_config, "max_tokens") else 1024,
            )
            
            # 3. Call model via AsyncModelClient
            response = await client.chat_completions(request)
            
            # 4. Format result
            result = {
                "item_id": getattr(item, "id", "unknown"),
                "model_answer": response.choices[0].message.content,
                "prompt": user_prompt,
                "system_prompt": system_prompt,
                "latency_ms": response.performance.latency_ms if response.performance else None,
                "status": "success"
            }
            # Merge with original item data for reporting
            result.update(context)
            return result
            
        except Exception as e:
            logger.error(f"Error processing item {getattr(item, 'id', 'unknown')}: {str(e)}")
            return {
                "item_id": getattr(item, "id", "unknown"),
                "status": "error",
                "error": str(e)
            }

async def run_async_vlm():
    # 1. Setup configurations using v0.2.0 Registry
    registry_root = os.getenv("REGISTRY_ROOT", "/workspace/vlmhyperbench/registries")
    run_name = os.getenv("RUN_NAME")
    
    if not run_name:
        logger.error("RUN_NAME environment variable not set")
        return

    registry = RegistryManager(root_dir=registry_root, run_mode=os.getenv("RUN_MODE", "dev"))
    run_config = registry.get_run(run_name)
    
    # 2. Initialize PromptManager
    # In v0.2.0, prompt collections are stored in data_dirs/PromptCollections/{task_name}/default.json
    data_root = os.getenv("DATA_ROOT", "/workspace/data")
    prompt_config_path = os.path.join(
        data_root, "PromptCollections",
        run_config.task,
        "default.json"
    )
    prompt_manager = PromptManager.from_json(prompt_config_path)
    
    # 3. Initialize AsyncModelClient
    api_url = os.getenv("API_URL", "http://localhost:8000")
    auth_token = os.getenv("AUTH_TOKEN")
    client = AsyncModelClient(base_url=api_url, auth_token=auth_token)
    
    try:
        # 4. Load model in API Wrapper
        load_request = ModelLoadRequest(
            model_name=run_config.model_name,
            backend_class="vllm_adapter.adapter:vLLMAdapter", # Match the actual class name in adapter.py
            engine_params={
                "cache_dir": "/workspace/model_cache",
                # Add other params as needed
            }
        )
        await client.load_model(load_request)
        logger.info(f"Model {run_config.model_name} loaded successfully.")

        # 5. Initialize Dataset Iterator
        dataset_config = registry.get_dataset(run_config.dataset)
        dataset_dir_path = os.path.join("/workspace", dataset_config.path)
        
        iterator = DatasetFactory.create(
            type_name=dataset_config.iterator_class,
            dataset_dir_path=dataset_dir_path,
        )
        
        # 6. Run Inference Loop with Semaphore
        concurrency = int(os.getenv("CONCURRENCY", 4))
        semaphore = asyncio.Semaphore(concurrency)
        
        tasks = [
            process_item(semaphore, client, item, prompt_manager, run_config)
            for item in iterator
        ]
        
        logger.info(f"Starting async inference for {len(tasks)} items with concurrency {concurrency}")
        results = await asyncio.gather(*tasks)
        
        # 7. Save results
        answers_dir = "/workspace/data/ModelsAnswers"
        os.makedirs(answers_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Clean model name for filename
        safe_model_name = run_config.model_name.replace("/", "_")
        output_filename = f"answers_{safe_model_name}_{timestamp}.csv"
        output_path = os.path.join(answers_dir, output_filename)
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    finally:
        await client.unload_model()
        await client.close()

if __name__ == "__main__":
    asyncio.run(run_async_vlm())
