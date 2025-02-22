# TaskWeaverRemake

一个基于大语言模型的多智能体数据分析系统，支持自然语言任务规划和代码自动生成执行。
参考taskweaver项目，进行了简化复现。

## 项目架构

本项目包含以下核心组件:

- **BaseAgent**: 智能体基类，提供LLM调用、案例检索等基础功能
- **Planner**: 负责理解用户任务并制定执行计划
- **CodeInterpreter**: 负责生成和执行Python代码
- **EnvironmentContextManager**: 管理多智能体之间的对话上下文
- **CaseRetriever**: 基于向量数据库的相似案例检索系统

系统工作流程:

1. 用户输入自然语言任务描述
2. Planner分析任务并制定执行计划
3. CodeInterpreter根据计划生成并执行代码
4. 根据执行结果继续规划或完成任务

## 环境配置

本项目使用uv进行环境管理。首先安装uv:

```bash
pip install uv
```

拉取项目并配置环境

```bash
git clone https://Sisicca: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@github.com/Sisicca/TaskWeaverRemake.git
```

配置环境（运行hello.py会自动同步依赖）

```bash
cd TaskWeaverRemake
uv run hello.py
```

激活环境

```bash
source .venv/bin/activate
```

## 配置config文件

在config文件夹下创建config.yaml文件，并根据config_template.yaml文件配置。

配置文件说明：

- **planner配置**:
  - alias: 智能体别名
  - openai_api_key: OpenAI API密钥
  - openai_base_url: OpenAI API基础URL
  - model: 使用的语言模型(推荐gpt-4o)
  - model_kwargs: 模型参数配置
    - temperature: 采样温度
  - embedding_model: 使用的嵌入模型
  - cases_path: 任务规划案例路径
  - max_workers: 调用嵌入模型最大工作线程数
  - workspace_path: 任务规划工作空间(需与code_interpreter一致)。当前任务数据文件等需存储于此。

- **code_interpreter配置**:
  - alias: 智能体别名
  - openai_api_key: OpenAI API密钥
  - openai_base_url: OpenAI API基础URL
  - model: 使用的语言模型(推荐gpt-4o)
  - model_kwargs: 模型参数配置
    - temperature: 采样温度
  - embedding_model: 使用的嵌入模型
  - cases_path: 代码执行案例路径
  - max_workers: 调用嵌入模型最大工作线程数
  - workspace_path: 代码执行工作空间(需与planner一致)。当前任务数据文件等需存储于此。
  - notebook_name: 代码执行jupyter文件名

## 运行项目

```bash
uv run -m src.main --config config/config.yaml
```

## 输出结果

- **代码执行日志**: 在code_interpreter的workspace_path路径下生成notebook文件，记录代码执行日志。
- **智能体交互日志**: src/logs下，记录智能体交互日志。


