# 可视化数据标注器使用说明

## 功能概述

这个工具用于标注 `edited_raw_all.jsonl` 中的数据，通过图形界面显示每条记录的元数据和对应的编辑后图片，支持快速的 Yes/No 标注。

## 运行方式

```bash
cd /gpfs/projects/p32958/chengxuan/ProgressLM/data/utils_img/visual_nega
python visual_annotation_tool.py
```

## 界面说明

### 布局结构

```
┌─────────────────────────────────────────────────────────────────┐
│  记录 1/6377  (已标注: ✓ YES)                                    │
├────────────────────────────┬────────────────────────────────────┤
│  数据信息                  │  编辑后的图片                       │
│  ────────────────────      │                                    │
│  STRATEGY                  │                                    │
│  Occlusion/Removal         │        [图片显示区域]               │
│                            │                                    │
│  PROMPT                    │                                    │
│  Remove the green plate... │                                    │
│                            │                                    │
│  RAW DEMO                  │                                    │
│  [left] approaches...      │                                    │
│                            │                                    │
│  META DATA                 │                                    │
│  - task_goal: ...          │                                    │
│  - image: camera_...jpg    │                                    │
│  - id: h5_agilex_3rgb/...  │                                    │
│  - data_source: ...        │                                    │
│  - status: success         │                                    │
│  - text_demo:              │                                    │
│    1. [left] approaches... │                                    │
│    2. [left] grab...       │                                    │
│    ...                     │                                    │
├────────────────────────────┴────────────────────────────────────┤
│  [← 上一条] [✓ YES] [✗ NO] [跳过 →]    已标注: 50 | YES: 30 | NO: 20  [保存并退出]  │
└─────────────────────────────────────────────────────────────────┘
```

### 控制按钮

- **← 上一条**: 返回上一条记录（快捷键: ← 左箭头）
- **✓ YES**: 标注为保留，并自动跳到下一条（快捷键: Y）
- **✗ NO**: 标注为删除，并自动跳到下一条（快捷键: N）
- **跳过 →**: 不标注，直接跳到下一条（快捷键: → 右箭头）
- **保存并退出**: 保存标注结果并退出程序（快捷键: Ctrl+S）

### 快捷键列表

| 快捷键 | 功能 |
|--------|------|
| Y | 标注为 YES（保留） |
| N | 标注为 NO（删除） |
| ← | 上一条记录 |
| → | 下一条记录 |
| Ctrl+S | 保存并退出 |

## 功能特性

### 1. 自动进度保存

- 每次标注后自动保存进度到 `annotation_progress.json`
- 程序关闭后重新打开会从上次位置继续
- 可以随时中断标注，不会丢失进度

### 2. 图片路径处理

工具会自动将图片名称转换为编辑后的版本：
- 原始图片名: `camera_front_0368.jpg`
- 编辑后图片: `camera_front_0368_edit.jpg`
- 完整路径: `/gpfs/projects/p32958/chengxuan/results/progresslm/negative/image/{id}/camera_front_0368_edit.jpg`

### 3. 图片缺失处理

如果某条记录的图片不存在，界面会显示错误信息：
```
图片不存在:
/path/to/image/camera_front_0368_edit.jpg
```
你仍然可以继续标注其他记录。

### 4. 实时统计

底部状态栏实时显示：
- 已标注总数
- YES（保留）数量
- NO（删除）数量

### 5. 可重新标注

- 已标注的记录可以重新浏览和修改
- 顶部会显示该记录的当前标注状态
- 重新标注会覆盖之前的选择

## 输出文件

### 1. annotated_output.jsonl

包含所有标注为 **YES** 的记录，格式与原始文件相同：
```json
{"strategy": "Occlusion/Removal", "prompt": "...", "raw_demo": "...", "meta_data": {...}}
```

### 2. annotated_output_stats.txt

标注统计信息：
```
标注统计信息
==================================================
总记录数: 6377
已标注数: 6377
YES (保留): 4520
NO (删除): 1857
未标注: 0
保留率: 70.88%
```

### 3. annotation_progress.json

进度文件（标注完成后会自动删除）：
```json
{
  "current_index": 1234,
  "annotations": {
    "0": true,
    "1": false,
    "2": true,
    ...
  }
}
```

## 数据显示说明

### 显示的字段

1. **STRATEGY**: 视觉破坏策略（如 Occlusion/Removal, Color Change 等）
2. **PROMPT**: 编辑指令，描述如何修改图片
3. **RAW DEMO**: 原始演示文本（机器人动作指令）
4. **META DATA**: 完整的元数据信息
   - `task_goal`: 任务目标描述
   - `image`: 原始图片文件名
   - `text_demo`: 多步演示动作序列（通常10+步骤）
   - `id`: 轨迹标识符
   - `data_source`: 数据来源
   - `status`: 处理状态

### text_demo 示例

```
text_demo:
  1. [left] approaches the green plate while [right] grabs the green plate
  2. [left] grab the green plate while [right] away from the green plate
  3. [left] move the green plate towards the plate rack
  4. [left] put the green plate in the plate rack
  5. [left] away from the green plate while [right] move towards the light brown plate
  ...
```

## 常见问题

### Q1: 图片加载失败怎么办？

**A:** 检查图片路径是否正确：
1. 确认 `meta_data.id` 和 `meta_data.image` 字段正确
2. 检查图片目录是否存在
3. 确认编辑后的图片文件名格式为 `*_edit.jpg`

### Q2: 如何恢复之前的标注进度？

**A:** 只要 `annotation_progress.json` 文件存在，程序会自动加载并恢复进度。

### Q3: 可以修改已标注的记录吗？

**A:** 可以！使用 ← 和 → 箭头键浏览所有记录，重新标注会覆盖之前的选择。

### Q4: 如何只保存部分标注结果？

**A:** 程序只保存标注为 YES 的记录到输出文件。未标注的记录不会被保存。

### Q5: 标注到一半需要休息，怎么办？

**A:** 直接关闭窗口即可，进度会自动保存。下次打开程序会从上次位置继续。

## 依赖环境

```bash
# 需要的Python包
pip install pillow  # 图片处理库
```

tkinter 是 Python 内置库，无需额外安装。

## 建议工作流程

1. **首次运行**: 从第1条记录开始标注
2. **快速标注**: 使用 Y/N 快捷键提高效率
3. **定期保存**: 每标注100-200条可以按 Ctrl+S 保存一次（可选）
4. **分批处理**: 可以分多次完成，每次标注一部分
5. **最终检查**: 完成后可以再浏览一遍，检查是否有误标

## 性能说明

- 数据加载: 约1-2秒（6377条记录）
- 图片加载: 实时加载，每张约0.1-0.3秒
- 标注响应: 即时响应，无延迟
- 内存占用: 约100-300MB（取决于图片大小）

## 技术支持

如有问题，请检查：
1. Python 版本 >= 3.6
2. Pillow 库已正确安装
3. 文件路径权限正确
4. 图片文件存在且可访问
