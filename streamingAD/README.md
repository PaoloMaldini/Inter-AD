# StreamingAD — 流式音频描述生成

基于 Video-LLaMA 模型的实时音频描述（Audio Description）生成系统。采用 Gradio 交互界面，用户拖动时间轴即可获得任意时刻的电影 AD 文本。

---

## 1. 项目概述

### 1.1 背景

newAD 项目（`Step04_RunTest/`）已经实现了离线批量 AD 生成：读取预处理好的 segment 数据（场景、人脸、剧本），对每个 AD 片段调用 Video-LLaMA 模型生成一句画面描述。本项目的目标是将这一能力包装为**交互式流式界面**，用户可以通过 Gradio UI 拖动时间轴，实时查看任意时间点的 AD 生成结果。

### 1.2 目录结构

```
streamingAD/
├── README.md              # 本文档
├── run.sh                 # 启动脚本（设置环境变量 + 启动 Gradio）
├── streaming_ad.py        # Gradio 界面层（UI + 事件处理）
├── ad_engine.py           # 模型推理引擎（加载 Video-LLaMA + 单段推理）
├── context_builder.py     # Prompt 构建器（Plot / Character / Scene 上下文 + Task 指令）
└── segment_db.py          # 片段数据库（加载 segment JSON + 时间查询 + 人脸数据提取）
```

---

## 2. newAD 原有思路

### 2.1 核心链路

```
step04_final_by_movie/*.json
  └─→ ad_segments[]  (每个 segment 包含: cmdqa, aggregated, clip_index, ...)

step04_03_face_align/json/<movie>/<clip>_ad<ad>.json
  └─→ detections[].match  (角色名称 + 人脸图片路径)

step04_RunTest/ad_clips/<movie>/<clip>_ad<ad>.mp4
  └─→ 已截取好的视频片段（2 秒左右）
```

对于每个 segment，`step04_04_imagehere.py` 执行以下步骤：

1. **构建 context**：Plot Database → character mapping（含头像上传） → scene info → `Target AD Clip:<Video><ImageHere></Video>`
2. **对话流**：
   - `chat.ask(context_text)` — 将上下文文本（含 `<ImageHere>` 占位符）写入对话
   - `chat.upload_img(face)` — 逐个上传角色人脸（每个 `upload_img` 在 messages 中追加 `<Image><ImageHere></Image>` 并在 img_list 中 push embedding）
   - `chat.upload_video_without_audio(clip)` — 上传视频（8 帧均匀采样，在 messages 中追加 `<Video><ImageHere></Video>`，img_list push video embedding）
   - `chat.ask(task_prompt)` — 追加任务指令
   - `chat.answer()` — 推理生成 AD 文本
3. **输出**：每个 segment 的状态和生成的 AD 文本写入结果 JSON

### 2.2 关键机制：`<ImageHere>` 占位符

Video-LLaMA 使用 `<ImageHere>` 作为 visual embedding 的文本占位符。在 `get_context_emb()` 中：

```python
prompt_segs = prompt.split('<ImageHere>')
assert len(prompt_segs) == len(img_list) + 1
```

它将完整 prompt 按 `<ImageHere>` 分割，然后在每个分割点之间插入 img_list 中对应的视觉 embedding。因此 **prompt 中 `<ImageHere>` 的数量必须严格等于 img_list 中 embedding 的数量**，否则直接断言失败。

### 2.3 `ask()` 的合并机制

```python
def ask(self, text, conv):
    if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
            and ('</Video>' in conv.messages[-1][1] or '</Image>' in conv.messages[-1][1]):
        conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
    else:
        conv.append_message(conv.roles[0], text)
```

**当上一条消息包含 `</Video>` 或 `</Image>` 时，新文本不会新建消息，而是直接追加到上一条消息的末尾。** 这意味着 `upload_video_without_audio` 之后立刻 `ask(task_prompt)` 时，task_prompt 会和视频占位符合并在同一条 Human 消息中。

---

## 3. streamingAD 设计思路

### 3.1 架构原则

- **推理方法与 newAD 完全一致**：模型加载（Config / registry / Chat）、对话流（context → upload_video → ask → answer）、视觉处理器配置，全部与 `step04_04_imagehere.py` 相同。
- **Prompt 上下文结构保持一致**：Plot Database → character mapping → scene info → Target AD Clip（仅 Format 做了适配性调整，见第 4 节）。
- **UI 层仅为薄壳**：Gradio 层仅负责界面渲染和事件转发，所有领域逻辑在独立模块中实现。

### 3.2 模块职责

| 模块 | 职责 |
|------|------|
| `segment_db.py` | 加载 step04 预处理数据（segment 信息、人脸匹配），提供按时间查询 segment 的接口 |
| `context_builder.py` | 按 newAD 的顺序构建 prompt：plot → character → scene → Target AD Clip；构建 task prompt（支持自定义指令） |
| `ad_engine.py` | 封装 Video-LLaMA 模型加载和单段推理（装箱 init + `infer_one_segment`） |
| `streaming_ad.py` | Gradio UI：时间轴控制、指令输入、温度调节、生成结果展示 |

### 3.3 数据流

```
用户在 Gradio 拖动时间轴
  → segment_db.current_segment(time_sec)  找到对应 segment
  → context_builder.build_prompt_context() 构建上下文文本
  → context_builder.build_task_prompt(instruction) 构建任务指令
  → ad_engine.infer_one_segment(clip, context, task)
      1. chat.ask(context)
      2. chat.upload_video_without_audio(clip)
      3. chat.ask(task)
      4. chat.answer()
  → 返回 (ad_text, elapsed, raw_prompt)
  → Gradio chatbot 展示结果；终端打印完整 prompt
```

---

## 4. 从 newAD 到 streamingAD 的关键差异与修改

### 4.1 `<ImageHere>` 占位符处理

**问题**：newAD 在 `build_prompt_context()` 末尾加了 `Target AD Clip:<Video><ImageHere></Video>`，纯文本中的 `<ImageHere>` 会被计入 prompt 占位符计数，而它没有对应的 img_list embedding。加上 `upload_video_without_audio` 又注入一个带 embedding 的 `<ImageHere>`，导致 prompt 中 2 个占位符 vs img_list 1 个元素 → **AssertionError: Unmatched numbers of image placeholders and images**。

**解决**：streamingAD 的 `build_prompt_context()` 移除了末尾的 `<Video><ImageHere></Video>`，改为纯文本 `Target AD Clip:`。`<ImageHere>` 由 `upload_video_without_audio` 自动注入，确保精确匹配：prompt 中 1 个 `<ImageHere>` ⇔ img_list 中 1 个 video embedding。

> 验证：对 Shawshank Redemption 5 个 segment 的端到端测试中，每个 segment 的 `ImageHere count = 1`，与 `img_list` 长度完全匹配，零断言错误。

### 4.2 Task Prompt 格式

**问题**：newAD 使用的多行结构化 prompt（`[Role]...[Operational Instructions]...[Output Template]`）在此模型上效果很差：

- `"[Role]\nYou are a Cinematic AD Specialist...\n[Operational Instructions]...\n[Output Template]..."` → 模型输出 `"Hello."`
- `"What's the scene like?"` 等无关内容

单行指令格式则正常工作：

- `"Describe what is happening..."` → `"The man turns to the other inmate and nods."`

**解决**：改为简洁单行指令：

```
Describe what is happening in this clip concisely.
Focus on visible actions, movements, and expressions.
Do not quote character names or dialogue.
{自定义指令占位符}
```

> 这是 streamingAD 与当前版本 newAD step04_04 的最大分歧点。当前 step04_04 脚本在此环境（conda videollava）下使用相同的结构化 prompt **同样无法工作**——对 Shawshank 第一个 segment 直接触发了 `<ImageHere>` 数量不匹配的断言错误（因为 context 中包含了多余的 `<ImageHere>`）。

### 4.3 单段推理 vs 批量处理

| 维度 | newAD (step04_04) | streamingAD |
|------|-------------------|-------------|
| 推理模式 | 批量遍历所有 segments | 用户触发式，每次 1 个 segment |
| 人脸上传 | 按角色逐一 `upload_img(face)` | 当前通过 context 中 `<rolename>` 标签传递角色信息（待接入 `upload_img`） |
| 超时机制 | 有 `signal.SIGALRM` 超时保护 | 无超时（用户交互式，不需要） |
| 结果缓存 | 写 JSON 文件，支持 `--overwrite` | 无持久化，仅展示在 chatbot 中 |

### 4.4 Gradio 兼容性修复

**问题**：Gradio 3.24.1 的 `PredictBody` Pydantic 模型缺少 `event_id` 字段，导致前端请求返回 422 Unprocessable Entity。

**解决**：在 `streaming_ad.py` 顶部 monkey-patch `Queue.get_message`，在接收到的 JSON 中补上缺失的 `event_id` 字段：

```python
async def _patched_get_message(self, event, timeout=5):
    data = await asyncio.wait_for(event.websocket.receive_json(), timeout=timeout)
    data.setdefault('event_id', '')
    return PredictBody(**data), True
```

---

## 5. 5 段测试结果验证

对 Shawshank Redemption 的端到端测试（conda videollava 环境，GPU 1）：

| Seg | newAD 原结果 | streamingAD 结果 |
|-----|-------------|-------------------|
| 0 | `tries to stand up.` | `The man turns to the other inmate and nods.` |
| 1 | `tries to stand up.` | `Quartz?` |
| 2 | `tightens his mouth.` | `The police officers stand at attention.` |
| 10 | `He looks at him with a mixture of sadness and contempt.` | `The man's eyes are fixed on the other man's face.` |
| 30 | `together.` | `The two boys sit on the ground outside the prison wall.` |

所有 5 段均未触发 `<ImageHere>` 占位符不匹配错误，推理速度约 1-3 秒/段。

---

## 6. 启动方式

```bash
cd /mnt/disk1new/ylz/newAD
conda activate videollava
export GPU_ID=1 HF_HOME=/tmp/hf_cache TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
bash streamingAD/run.sh
```

Gradio 界面运行在 `http://0.0.0.0:7860`。可通过环境变量控制：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GPU_ID` | 0 | GPU 编号 |
| `PORT` | 7860 | Web 端口 |
| `MOVIE_PATH` | Shawshank | 电影文件路径 |
| `SHARE` | (空) | 设为 1 生成公网链接 |

---

## 7. 依赖与环境

- **Conda 环境**：`/mnt/disk6new/wzq/env/videollava`
- **Python**：3.10.18
- **PyTorch**：2.0.1+cu118
- **GPU**：NVIDIA RTX A6000
- **模型文件**：
  - `/mnt/disk1new/ylz/newAD/models/llama-2-7b-chat-hf`
  - `/mnt/disk1new/ylz/newAD/models/imagebind_huge.pth`
  - `/mnt/disk1new/ylz/newAD/models/ad3_moviellama2_ce_iter14000.pth.tar`
- **数据目录**：
  - Segment: `/mnt/disk1new/ylz/newAD/Step04_RunTest/step04_final_by_movie_new/`
  - 人脸: `/mnt/disk1new/ylz/newAD/Step04_RunTest/step04_03_face_align/json/`
  - 视频片段: `/mnt/disk1new/ylz/newAD/Step04_RunTest/ad_clips_final/`
