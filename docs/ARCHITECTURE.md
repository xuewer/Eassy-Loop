# Eassy Loop 架构

定位：面向英文短文的多维质量评测与反馈优化框架，使用进程内 LangGraph 多角色节点，不包含标准 A2A 协议。

~~~mermaid
flowchart TD
    S([START]) --> M{mode}
    M -->|optimize| G[初次生成或反馈修订]
    M -->|evaluate| D[dispatch]
    G --> F{成功或已有初稿}
    F -->|失败| E([END])
    F -->|成功| D
    D --> R[relevance]
    D --> A[grammar]
    D --> T[structure]
    D --> P[depth]
    R --> B[四分支汇聚屏障]
    A --> B
    T --> B
    P --> B
    B --> Q[校验轮数和文章哈希并聚合]
    Q --> C{条件路由}
    C -->|失败或达到退出条件| E
    C -->|继续| G
~~~

串行对照：dispatch → relevance → grammar → structure → depth → aggregate。并行使用四个独立节点和显式多源边等待全部分支；各裁判只写自己的状态字段。

## 状态与不变量

- optimize 初始轮数为 0，进入 generator 后开始第一轮；传入初稿时先评测，不额外生成。evaluate 从第一轮开始，评分一次结束。
- 每轮开始清空四维结果和 final_score，避免旧轮污染。四维必须全部成功、round_id 和 essay_hash 匹配才能聚合。
- 反馈包含评分、理由、证据和建议，按低分排序；all/relevance/none 只改变重写器收到的反馈。none 仍要求自行改进文章。
- history 保存各轮文章和裁判结果；generation_history 保存实际反馈、输入输出哈希及尝试。哈希变化只证明文本变化。
- threshold 策略检查总分和可选四维最低分；fixed 只按轮数停止。失败立即停止，不填默认分数，不回退历史最高分。
- Recorder 用锁保护调用记录；客户端 usage 使用线程局部存储，避免四路并发串数据。

## 调用与计时

已有初稿且无重试：evaluate 为 4 次；R 轮 optimize 为 4R+(R−1)=5R−1 次；无初稿时额外首次生成一次，为 5R 次。保守上界再乘 max_attempts。

calls.jsonl 的 status 表示模型调用结果，解析失败记录在节点 attempts；parse_failures 包含最终成功前的失败尝试。

elapsed_seconds 包含客户端初始化、图执行和重试等待，不含最终写盘与指纹收集；latency_seconds 包围模型调用，不等于纯服务端推理时间。四路并行的实际收益取决于限流、网络和服务负载，需实测。

## 复现边界

固定 seed 控制计划顺序和 bootstrap，不控制远端模型随机性。manifest 记录 Python、依赖版本、Git HEAD 和源码 SHA256；未提交代码不能仅依赖 HEAD 复现，交付时需对应源码快照和数据授权说明。
