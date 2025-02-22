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


