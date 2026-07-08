# foundry-gcp-databricks-integration-demo

Azure 側から Google Cloud 上の Databricks データを活用するための検証メモです。

主な検証観点は次の 4 点です。

1. 最新データを必要なタイミングで Databricks から直接参照する
2. Databricks のデータ更新に合わせて Azure AI Search などのベクトル DB を自動更新する
3. Databricks のデータを利用して定期的に AI モデルを再学習する
4. 同じ Databricks データを複数の Agent やアプリケーションから利用できるようにする

詳細な検証手順は [samples/手順書.md](samples/手順書.md) を参照してください。

## サンプル

> **制限事項**: 検証 1 の OpenAPI Tool は SQL statement の送信操作のみ定義しています。クエリが `PENDING` または `RUNNING` を返した場合のポーリング操作は含まれていません。短時間で完了する単純な集計クエリで動作確認してください。長時間クエリが必要な場合は Databricks の get-statement 操作を追加してください。

- [samples/databricks-tool-openapi.json](samples/databricks-tool-openapi.json): Foundry Agent から Databricks Statement Execution API を呼び出す OpenAPI Tool 定義
- [samples/deploy_foundry_openapi_agent.py](samples/deploy_foundry_openapi_agent.py): OpenAPI Tool 付き Foundry Agent version を作成するスクリプト
- [samples/test_databricks_statement_api.py](samples/test_databricks_statement_api.py): Databricks Statement Execution API の直接疎通確認スクリプト
- [samples/sync_databricks_to_ai_search.py](samples/sync_databricks_to_ai_search.py): Databricks の行を embedding 化して Azure AI Search に upsert する検証 2 用スクリプト
- [samples/sync_databricks_cdf_to_ai_search.py](samples/sync_databricks_cdf_to_ai_search.py): Databricks Delta Change Data Feed の差分を Azure AI Search に反映する検証 2 用スクリプト
- [samples/train_sales_model_from_databricks.py](samples/train_sales_model_from_databricks.py): Databricks の売上データからモデル artifact と metrics を作成する検証 3 用スクリプト
- [samples/verify_sales_model_artifacts.py](samples/verify_sales_model_artifacts.py): 検証 3 のモデル artifact と metrics が作成済みか確認するスクリプト
- [azureml/sales-retrain-job.yml](azureml/sales-retrain-job.yml): 検証 3 を Azure ML command job として実行するための定義
- [azureml/sales-retrain-schedule.yml](azureml/sales-retrain-schedule.yml): 検証 3 の Azure ML 定期実行 schedule 定義
- [azureml/submit-sales-retrain-job.ps1](azureml/submit-sales-retrain-job.ps1): Databricks PAT を非表示入力して Azure ML command job を投入する PowerShell スクリプト
