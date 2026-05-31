"""新质生产力语义量化与耦合协调分析的核心算法库。

模块划分：

- ``nqp.index``         熵权法、min-max 归一化、综合指数合成
- ``nqp.coupling``      耦合度 C、耦合协调度 D 及其等级划分
- ``nqp.textfeatures``  中文分词、TF-IDF、关键词共现
- ``nqp.panel``         面板数据构造与双向固定效应回归的封装

脚本目录 ``code/0X_*`` 只做命令行封装，真正的数值逻辑都在这里，便于单元测试。
"""

from nqp import coupling, index

__all__ = ["index", "coupling"]
__version__ = "0.2.0"
