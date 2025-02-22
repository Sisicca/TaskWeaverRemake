from abc import ABC, abstractmethod
import faiss
import yaml
import numpy as np
import json
import os
from typing import List, Dict, Any
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from jsonschema import validate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CaseRetriever:
    def __init__(self, config: Dict[str, Any]):
        self.client = OpenAI(api_key=config['openai_api_key'], base_url=config['openai_base_url'])
        self.embedding_model = config['embedding_model'] # 用于embedding的模型
        self.cases_path = config['cases_path'] # 用例路径
        self.max_workers = config['max_workers'] # 用于embedding的线程数
        
        self.index = None
        # 验证cases_path是否存在
        if not os.path.exists(self.cases_path):
            raise FileNotFoundError(f"Cases path does not exist: {self.cases_path}")
        if not os.path.isdir(self.cases_path):
            raise NotADirectoryError(f"Cases path is not a directory: {self.cases_path}")
        # 初始化index
        try:
            self._create_index()
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    def _get_single_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=text, model=self.embedding_model)
        return response.data[0].embedding
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._get_single_embedding, text): index
                for index, text in enumerate(texts)
            }
            embeddings = [None] * len(texts)
            for future in as_completed(futures):
                try:
                    index = futures[future]
                    embeddings[index] = future.result()
                except Exception as e:
                    logger.error(f"Error embedding text {index}: {e}")
        return embeddings
    
    def _create_index(self):
        self.cases = []
        descriptions = []
        yaml_files = [f for f in os.listdir(self.cases_path) if f.endswith('.yaml')]
        
        if not yaml_files:
            logger.warning(f"No YAML files found in {self.cases_path}")
            return
            
        for file_name in yaml_files:
            try:
                with open(os.path.join(self.cases_path, file_name), 'r', encoding='utf-8') as f:
                    case = yaml.safe_load(f)
                    if not isinstance(case, dict):
                        logger.warning(f"Invalid YAML format in {file_name}, skipping...")
                        continue
                    self.cases.append(case)
                    description = case.get('metadata', {}).get('description', 'No description')
                    descriptions.append(description)
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {str(e)}")
                continue

        if not descriptions:
            logger.warning("No valid descriptions found in cases")
            return

        try:
            embeddings = self._get_embeddings(descriptions)
            if not embeddings or not all(embeddings): # 如果embeddings为空或者有空值，则抛出异常
                raise ValueError("Failed to generate embeddings for all descriptions")
            # 创建faiss索引
            self.index = faiss.IndexFlatL2(len(embeddings[0]))
            # 将embeddings转换为numpy数组并添加到索引中
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
            logger.info(f"Index created successfully with {len(embeddings)} cases")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None:
            self._create_index()
        query_embedding = self._get_single_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.cases[i] for i in indices[0]]

class BaseAgent(ABC):
    """
    所有智能体的基类

    Args:
        config (Dict[str, Any]): 配置文件
    """
    def __init__(self, config: Dict[str, Any]):
        self.alias = config['alias']
        self.client = OpenAI(api_key=config['openai_api_key'], base_url=config['openai_base_url'])
        self.model = config['model']
        self.case_retriever = CaseRetriever(config)
        self.model_kwargs = config['model_kwargs']
        self.workspace_path = config['workspace_path']
        
        self.workspace_context = []
        # 验证workspace_path是否存在
        if not os.path.exists(self.workspace_path):
            raise FileNotFoundError(f"Workspace path does not exist: {self.workspace_path}")
        if not os.path.isdir(self.workspace_path):
            raise NotADirectoryError(f"Workspace path is not a directory: {self.workspace_path}")
        # 初始化workspace_context
        self._update_workspace_context()
    
    def _decorate_response(self, response: Dict[str, Any], role: str, send_to: str | None = None) -> Dict[str, Any]:
        final_response = {"role": role} 
        final_response.update(response)
        if send_to:
            final_response.update({"send_to": send_to})
        return final_response
    
    def _update_workspace_context(self):
        """
        更新workspace_path下的所有文件列表（包括子文件夹、空文件夹）
        确保返回完整的工作区路径+文件名的列表
        
        Example:
            workspace_path: src/workspaces/test
            workspace_context: ['src/workspaces/test/folder1/test.txt', 'src/workspaces/test/folder2']
        """
        self.workspace_context = []
        # logger.info(f"Updating workspace context from: {self.workspace_path}")
        
        for root, dirs, files in os.walk(self.workspace_path):
            # 处理文件
            for file_name in files:
                abs_file_path = os.path.join(root, file_name)
                # 使用os.path.join确保路径分隔符的一致性
                rel_path = os.path.join(self.workspace_path, os.path.relpath(abs_file_path, self.workspace_path))
                self.workspace_context.append(rel_path)
                
            # 处理目录
            for dir_name in dirs:
                abs_dir_path = os.path.join(root, dir_name)
                rel_dir_path = os.path.join(self.workspace_path, os.path.relpath(abs_dir_path, self.workspace_path))
                self.workspace_context.append(rel_dir_path)
        
        # logger.debug(f"Updated workspace context: {self.workspace_context}")
    
    @abstractmethod
    def reply(self, message: str) -> str:
        pass
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.case_retriever.search(query, top_k)
    
    def _check_response_json_schema(self, response: str, response_json_schema: str) -> bool:
        """
        检查response是否符合response_json_schema

        Args:
            response (str): 响应内容
            response_json_schema (str): 响应内容格式

        Returns:
            bool: 是否符合响应内容格式
        """
        try:
            response_json = json.loads(response.strip("```json").strip("```"))
            schema = json.loads(response_json_schema)
            validate(instance=response_json, schema=schema)
            return True
        except Exception as e:
            logger.info(f"Invalid response_json: {response_json}")
            logger.error(f"Invalid response: {e}")
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    def _call_llm_for_json(self, messages: List[Dict[str, str]], response_json_schema: str = None) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **self.model_kwargs
        ).choices[0].message.content
        if response_json_schema:
            if not self._check_response_json_schema(response, response_json_schema):
                raise ValueError(f"Invalid response: {response}")
        return json.loads(response.strip("```json").strip("```"))
    


if __name__ == "__main__":
    from rich import print
    import os
    config = {
        'alias': 'test_agent',
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'openai_base_url': os.getenv('OPENAI_BASE_URL'),
        'embedding_model': 'text-embedding-3-small',
        'cases_path': 'src/cases/plans',
        'model': 'gpt-4o-mini',
        'max_workers': 10,
        'model_kwargs': {
            'temperature': 0.5,
            'max_tokens': 1000,
            'top_p': 1,
            'frequency_penalty': 0,
        },
        'workspace_path': 'src/workspaces/analysis_ili'
    }
    
    class TestAgent(BaseAgent):
        def reply(self, message: str) -> str:
            return "Hello, world!"
    
    agent = TestAgent(config)
    print(f"I am {agent.alias}")
    print(agent.reply("你好"))
    print(agent.case_retriever.cases)
    print(agent.search("病历文本标注任务", 1))
    print(agent.workspace_context)
    