# 多跳复杂度分类器

独立 BERT/RoBERTa 微调服务，供主后端通过 HTTP 调用。模型权重和正式标注数据不提交仓库。

## 数据格式

每行一个 JSON：

```json
{"question":"A与B相比有什么区别？","label":"multi","features":{"comparison_flag":true,"logic_connector_flag":false,"bridge_entity_count":1,"entity_count":2,"token_length":12}}
```

`features` 可省略，训练和服务会根据问题重新提取。

## 生成种子数据和训练

```powershell
cd multihop_classifier
python seed_data.py
python train.py --data data/multihop_seed.jsonl --output models/complexity-classifier
```

种子数据仅用于启动标注，不应当直接作为正式模型依据。建议人工复核至少 500 条单跳/多跳中文业务问题，覆盖比较、桥梁实体、逻辑连接、长问题和多实体问题。

## 评估

```powershell
python evaluate.py --data data/multihop_test.jsonl --model models/complexity-classifier
```

训练脚本输出 Accuracy、Precision、Recall、F1、AUC 和混淆矩阵，多跳召回目标不低于 0.85。

## 推理服务

```powershell
$env:MULTIHOP_MODEL_DIR="models/complexity-classifier"
uvicorn app:app --host 0.0.0.0 --port 8010
```

接口：`GET /health`、`POST /predict`、`POST /predict-batch`。
