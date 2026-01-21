import os
import asyncio
import logging
import json
import time
import pandas as pd
from typing import List, Dict, Any

from benchmark_run_config.benchmark_run_config import BenchmarkRunConfig
from config_manager.config_manager import ConfigManager
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
    run_config: BenchmarkRunConfig,
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
    # 1. Setup configurations
    cfg_path = os.getenv("VLMHYPERBENCH_CONFIG_PATH")
    if not cfg_path:
        logger.error("VLMHYPERBENCH_CONFIG_PATH not set")
        return

    cfg_container = ConfigManager(cfg_path).cfg_container
    run_config = BenchmarkRunConfig.from_json(cfg_container["benchmark_run_cfg"])
    
    # 2. Initialize PromptManager
    prompt_config_path = os.path.join(
        cfg_container["data_dirs"]["prompt_collections"], 
        run_config.task_name, 
        run_config.prompt_collection
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
            backend_class=f"{run_config.python_package}.{run_config.module}:{run_config.class_name}",
            engine_params={
                "cache_dir": cfg_container["system_dirs"]["model_cache"],
                # Add other params as needed
            }
        )
        await client.load_model(load_request)
        logger.info(f"Model {run_config.model_name} loaded successfully.")

        # 5. Initialize Dataset Iterator
        dataset_dir_path = os.path.join(
            cfg_container["data_dirs"]["datasets"], run_config.task_name, run_config.dataset
        )
        
        iterator = DatasetFactory.create(
            type_name=run_config.dataset_type if hasattr(run_config, "dataset_type") else run_config.task_name,
            dataset_dir_path=dataset_dir_path,
            # Add other filters
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
        answers_dir = cfg_container["data_dirs"]["model_answers"]
        os.makedirs(answers_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"answers_{run_config.model_name}_{timestamp}.csv"
        output_path = os.path.join(answers_dir, output_filename)
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Update run_config
        run_config.metric_file = output_path
        run_config.to_json(cfg_container["benchmark_run_cfg"])

    finally:
        await client.unload_model()
        await client.close()

if __name__ == "__main__":
    asyncio.run(run_async_vlm())
