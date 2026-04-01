# loan-default-xgboost

## Português

`loan-default-xgboost` é um projeto de classificação binária para predição de inadimplência, com `XGBoost` como modelo principal e `Logistic Regression` como baseline comparativo.

### Storytelling técnico: por que XGBoost aqui

Em problemas reais de crédito, o desafio raramente é apenas “prever default”. O problema central é ordenar risco com boa capacidade discriminativa, capturando interações não lineares entre renda, endividamento, utilização, histórico e comportamento recente. É exatamente nesse tipo de dado tabular que `XGBoost` ganhou espaço na prática.

`XGBoost` é uma implementação altamente otimizada de gradient boosting sobre árvores de decisão. Na prática, ele combina vários estimadores fracos em sequência, reduzindo o erro residual a cada nova árvore. Isso o torna especialmente forte quando:

- as relações entre variáveis não são lineares;
- existem interações difíceis de capturar com modelos lineares;
- o problema exige bom desempenho em dados tabulares;
- é importante manter um modelo competitivo sem partir para arquiteturas muito complexas.

Por isso este projeto foi desenhado como um case de risco de crédito em que:

- `Logistic Regression` estabelece um baseline linear interpretável;
- `XGBoost` entra como modelo principal para capturar mais estrutura no espaço tabular;
- um fallback com `HistGradientBoostingClassifier` mantém o projeto executável mesmo sem a dependência nativa do `xgboost`.

### Objetivo técnico

O foco é mostrar um caso forte de ML tabular com:

- feature engineering orientada a risco de crédito;
- baseline linear para comparação;
- `XGBoost` como modelo principal;
- métricas adequadas para classificação binária;
- inspeção de importância de atributos.

### Topologia do projeto

- [src/data_factory.py](src/data_factory.py)
  Gera um dataset sintético reproduzível com semântica de risco de crédito.
- [src/modeling.py](src/modeling.py)
  Executa o split de dados, treina baseline e modelo principal, calcula métricas e gera importâncias.
- [main.py](main.py)
  Entry point do projeto, responsável por persistir o relatório consolidado.
- [tests/test_project.py](tests/test_project.py)
  Valida a integridade do pipeline e a qualidade mínima do resultado.

### Modo de execução

O projeto prefere `XGBoost` quando a biblioteca está disponível no ambiente. Se `xgboost` não estiver instalado, o pipeline entra automaticamente em um fallback reproduzível com `HistGradientBoostingClassifier`, mantendo:

- execução local sem bloqueio;
- comparação contra baseline linear;
- ranking de importância por permutação.

### Estratégia de modelagem

O pipeline foi montado com três camadas:

1. `Baseline linear`
   `Logistic Regression` com `class_weight="balanced"` para estabelecer um referencial simples e interpretável.
2. `Modelo principal`
   `XGBoost`, quando disponível, para capturar interações não lineares e obter melhor ordenação do risco.
3. `Fallback reproduzível`
   `HistGradientBoostingClassifier`, usado quando `xgboost` não está instalado no ambiente.

Essa estrutura é útil porque evita dois extremos:

- depender apenas de um baseline fraco;
- bloquear a execução local por causa de uma dependência ausente.

### Stack

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `unittest`

### Pipeline

```mermaid
flowchart LR
    A["Synthetic credit data"] --> B["Feature engineering"]
    B --> C["Train/test split"]
    C --> D["Logistic Regression baseline"]
    C --> E["XGBoost model"]
    D --> F["Metric comparison"]
    E --> F
    E --> G["Feature importance"]
```

### Semântica das métricas

As métricas escolhidas refletem objetivos diferentes do problema:

- `ROC-AUC`
  Mede a capacidade de ordenação global entre bons e maus pagadores.
- `Average Precision`
  Dá uma leitura mais sensível à classe positiva, importante quando default é minoritário.
- `Precision`
  Indica quão “limpas” são as sinalizações de risco.
- `Recall`
  Mede cobertura sobre a classe de inadimplência.
- `F1`
  Resume o trade-off entre `precision` e `recall`.

Em um caso de underwriting, a leitura isolada de uma única métrica tende a ser insuficiente. Por isso o projeto preserva um bloco completo de avaliação para o baseline e para o modelo selecionado.

### Variáveis do dataset

O dataset sintético inclui sinais como:

- estabilidade de renda
- razão dívida/renda
- utilização de crédito
- atraso recente
- histórico de crédito
- volatilidade de caixa
- burden de parcelas
- interação entre utilização e delinquência

### Contrato do artefato

O relatório consolidado contém:

- `project_name`
- `dataset_rows`
- `positive_rate`
- `runtime_mode`
- `baseline_model`
- `baseline_metrics`
- `selected_model`
- `selected_model_metrics`
- `top_feature_importance`

Isso permite usar o projeto tanto como benchmark local quanto como bloco de partida para evoluções com serving, monitoramento e explicabilidade mais avançada.

### Artefato gerado

- `data/processed/loan_default_xgboost_report.json`

Esse arquivo é gerado em runtime e não é versionado.

### Execução

```bash
python3 main.py
python3 -m unittest discover -s tests -v
python3 -m py_compile main.py src/data_factory.py src/modeling.py
```

## English

`loan-default-xgboost` is a binary classification project for default prediction, using `XGBoost` as the primary model and `Logistic Regression` as a baseline.

When `xgboost` is not available in the runtime, the project automatically falls back to `HistGradientBoostingClassifier` so the pipeline remains executable and testable locally.

### Technical storytelling: why XGBoost matters here

In real credit risk problems, the challenge is not only to predict default, but to rank applicants with strong discriminative power while capturing non-linear interactions across income, debt pressure, utilization, and recent behavioral signals.

`XGBoost` is a highly optimized implementation of gradient boosting over decision trees. It is especially effective when:

- relationships across features are not linear;
- interaction effects matter;
- tabular performance is critical;
- a strong baseline is needed without moving to overly complex architectures.

This is why the project is structured around:

- `Logistic Regression` as an interpretable baseline;
- `XGBoost` as the preferred production-style model;
- `HistGradientBoostingClassifier` as a reproducible local fallback.

### Project topology

- [src/data_factory.py](src/data_factory.py)
  Deterministic synthetic credit dataset generation.
- [src/modeling.py](src/modeling.py)
  Modeling, evaluation, and feature-importance generation.
- [main.py](main.py)
  Consolidated entry point and runtime artifact writer.
- [tests/test_project.py](tests/test_project.py)
  Pipeline regression checks.

### Metric semantics

- `ROC-AUC`
  Global ranking quality across classes.
- `Average Precision`
  Positive-class sensitivity under class imbalance.
- `Precision`
  Signal cleanliness for predicted defaults.
- `Recall`
  Coverage over the default class.
- `F1`
  Precision/recall balance.
