# demo
套牌车识别项目版本管理
 # 日志
 ## 2025-11-12

- **[新增] 图像裁切预处理类**
  - 文件：`data_chuli/cropper.py`
  - 内容：新增 `VehicleCropper`，使用 YOLOv8 检测车辆并裁切最大目标，可选对车牌做黑色打码（HyperLPR3）。输入输出均为内存中的 `PIL.Image`。

- **[集成] GUI 预测前调用裁切**
  - 文件：`Siamese-pytorch-master/my_predict_gui.py`
  - 变更：导入并初始化 `VehicleCropper`，在 `predict_similarity` 中对两张图片先 `process_pil` 后再送入 Siamese 比对。

- **[修复] 概率格式化报错**
  - 文件：`Siamese-pytorch-master/my_predict_gui.py`
  - 变更：将 `detect_image` 的返回 `Tensor` 转为 `float` 再比较/格式化，避免 “unsupported format string passed to Tensor.__format__”。

- **[版本控制] 放开日志文件追踪**
  - 文件：`.gitignore`
  - 变更：新增 `!开发日志.md`，允许将开发日志提交到 Git。

## 2025-11-13

- **[新增] 批量套牌检测脚本**
  - 文件：`Siamese-pytorch-master/detect_clone_plates.py`
  - 功能：按车牌分组、仅向过去寻找“最近一趟”有图记录进行相似度比对；当前行无图=不可判定；仅使用 `过皮部位1图片URL`；默认阈值 `0.3`；输出 `clone_check_report.csv`。
  - 复用：`siamese.Siamese` 与 `data_chuli.cropper.VehicleCropper` 的预处理/推理流程。

- **[新增] 可视化批处理 GUI**
  - 文件：`Siamese-pytorch-master/clone_checker_gui.py`
  - 功能：选择 CSV、一键运行、展示当前/参考图片信息、相似度与判定结果；支持阈值调整；结果保存路径提示。
  - 复用：与命令行一致的模型与裁剪流程。

- **[修复] 导入路径**
  - 文件：`Siamese-pytorch-master/clone_checker_gui.py`
  - 变更：修正为从同目录导入 `detect_from_csv`，避免包路径错误。

- **[新增] 数据统计小工具**
  - 文件：`data_chuli/data_tongji.py`
  - 功能：按 `车号` 统计出现次数，输出 `plate_counts.csv` 与 `duplicate_rows.csv`，用于快速查看重复车牌与明细。

 
 ## 2026-01-05
 
 ### 阈值调整（疑似套牌）
 - **变更内容**：将“疑似套牌”头部相似度阈值默认值调整为 `0.8`。
 - **判定规则**：`head_prob < 0.8` 判定为疑似套牌（等于 `0.8` 不判定为低）。
 - **同步范围**：主批量检测逻辑 + “本地两图比对”工具页。
 
 ### 数据库批量检测：从指定日期开始（按 TASK_ID）
 - **入口**：“从数据库批量检测”弹窗新增模式 `从指定日期开始...`。
 - **日期选择**：日历选择起始日期（包含当天）。
 - **过滤规则**：按 `TASK_ID` 前 6 位 `yyMMdd` 解析日期，过滤 `TASK_YYMMDD >= 起始日期yyMMdd`。
 - **清理策略**：自动删除 `D:\output` 下起始日期之前的结果文件夹（按文件夹名中的 `TASK_ID` 前缀判断）。
 - **结果更新**：覆盖写入默认 CSV（无疑似结果时也覆盖为空表头，避免旧结果残留）。
 - **状态更新**：`last_task_id` 使用本次参与检测数据的最大 `TASK_ID` 更新。

 # 后端服务（Flask）

 ## 服务入口

 - 文件：`Siamese-pytorch-master/my_predict_gui_new1.py`
 - 框架：Flask
 - 功能：对两张车辆图片做预处理与裁切，然后分别计算“车头相似度/车尾相似度”，并给出简单分类结果。

 ## 核心流程（服务内部）

 - **车辆裁切预处理**：`data_chuli.cropper.VehicleCropper().process_pil()`
 - **部位裁切**：使用 YOLO 检测车头/车尾框（`ultralytics.YOLO`）
   - `cls_id=0`：车头
   - `cls_id=1`：车尾
 - **相似度模型**：分别用两套 Siamese 模型计算
   - 车头：`Siamese(model_path=HEAD_MODEL_PATH)`
   - 车尾：`Siamese(model_path=TAIL_MODEL_PATH)`
 - **并发控制**：初始化使用 `_INIT_LOCK`，推理使用 `_INFER_LOCK`，避免多线程并发导致模型状态异常。

 ## 接口说明

 ### `GET /`

 - 返回：可用 endpoints 列表

 ### `GET /health`

 - 返回：`{"status":"ok"}`

 ### `POST /predict`

 - **Content-Type**：`application/json`
 - **请求体**：
   - `path1`：图片1的绝对路径
   - `path2`：图片2的绝对路径
 - **路径校验规则**：
   - 必须是绝对路径
   - 文件必须存在且为图片格式（`.jpg/.jpeg/.png/.bmp/.webp`）
   - 如果设置了 `ALLOWED_BASE_DIRS`，则路径必须落在白名单目录内

 - **响应字段**：
   - `ok`：是否推理成功（`case_type != "abnormal"`）
   - `case_type`：分类结果（见下）
   - `head_prob`：车头相似度（float）
   - `tail_prob`：车尾相似度（float）
   - `error`：异常信息（可选）

 ## 分类规则（`case_type`）

 - `abnormal`
   - 模型初始化失败或推理异常（如文件打不开、模型路径错误等）
 - `fake_plate`
   - `head_prob < HEAD_LOW_TH`（默认 `0.8`）
 - `change_trailer`
   - `head_prob > HEAD_SAME_TH`（默认 `0.3`）且 `tail_prob <= TAIL_LOW_TH`（默认 `0.3`）
 - `normal`
   - 其余情况

 ## 环境变量配置

 - `HOST`
   - 默认：`0.0.0.0`
 - `PORT`
   - 默认：`8001`
 - `HEAD_MODEL_PATH`
   - 车头 Siamese 权重路径
   - 默认（脚本内置）：`Siamese-pytorch-master/logs/head/1211/best_epoch_weights.pth`
 - `TAIL_MODEL_PATH`
   - 车尾 Siamese 权重路径
   - 默认（脚本内置）：`Siamese-pytorch-master/logs/weibu/1211/best_epoch_weights.pth`
 - `HEADTAIL_MODEL_PATH`
   - YOLO 检测模型路径（用于裁切车头/车尾）
   - 默认（脚本内置）：`D:\data2\runs\detect\train\weights\best.pt`
 - `ALLOWED_BASE_DIRS`
   - 图片路径白名单；多个目录用英文分号 `;` 分隔
   - 示例：`D:\images;D:\dataset\capture`
 - `HEAD_LOW_TH` / `HEAD_SAME_TH` / `TAIL_LOW_TH`
   - 分类阈值，默认分别为 `0.8 / 0.3 / 0.3`

 ## 启动方式（Windows 示例）

 - 直接启动（使用脚本默认模型路径）：
  - `python Siamese-pytorch-master\my_predict_gui_new1.py`

 - 指定端口与模型路径（PowerShell）：
  - `$env:PORT="8001"; $env:HEAD_MODEL_PATH="D:\\path\\head.pth"; $env:TAIL_MODEL_PATH="D:\\path\\tail.pth"; $env:HEADTAIL_MODEL_PATH="D:\\path\\best.pt"; python Siamese-pytorch-master\my_predict_gui_new1.py`

 ## 调用示例

 - 请求：
```json
{
  "path1": "D:\\images\\a.jpg",
  "path2": "D:\\images\\b.jpg"
}
```

 - 响应示例：
```json
{
  "ok": true,
  "case_type": "normal",
  "head_prob": 0.91,
  "tail_prob": 0.88
}
