# Steam Price Predictor

Demonstração de ciência de dados que estima o preço histórico de jogos a partir de dados tabulares. O projeto compara configurações de `RandomForestRegressor` com `RandomizedSearchCV`, avalia o melhor estimador em um conjunto de teste separado e oferece uma interface Streamlit.

> A previsão é uma demonstração educacional baseada no dataset versionado. Não consulta a loja Steam e não deve ser interpretada como preço atual ou recomendação de compra.

## Requisitos

- Python 3.11
- Git

O dataset usado pelo exemplo está versionado em `data/steam.csv`; não é necessário obter uma chave de API ou baixar dados adicionais.

## Demonstração local reproduzível

Clone o repositório e execute os comandos a partir da pasta raiz:

```bash
git clone https://github.com/ZaraTakion/steam-price-predictor.git
cd steam-price-predictor
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Instale as dependências, treine o modelo com o ano de referência explícito e inicie a aplicação:

```bash
python -m pip install -r requirements.txt
python -m scripts.train --reference-year 2026
streamlit run app/app.py
```

Abra `http://localhost:8501`. O modelo treinado será salvo em `models/final_steam_model.pkl` e não precisa ser adicionado ao Git.

O ano passado em `--reference-year` é usado no cálculo `ano de referência - ano de lançamento`. O app lê esse ano salvo junto ao modelo, então treino e demonstração usam a mesma referência. Use o mesmo comando para reproduzir essa configuração em outras máquinas. Sem argumento, o script usa o ano atual; nesse caso, a feature de idade e os resultados podem mudar com o passar dos anos.

O ajuste faz 10 combinações aleatórias com validação cruzada de 5 partes e usa `random_state=42` no split, na busca e no Random Forest. A etapa de treino pode levar alguns minutos, dependendo do computador. Ao final, o terminal imprime os melhores parâmetros, R² e RMSE do teste; esses valores são medidos na execução e não são fixados como resultados garantidos.

## O que o treino mede

As features são estimativa média de proprietários, proporção e total de avaliações, log do tempo médio jogado, idade do jogo e gênero. O preço é o alvo. O app constrói exatamente as mesmas colunas antes de chamar o modelo.

## Testes

Os testes usam pequenos dados sintéticos e não executam a busca custosa de hiperparâmetros:

```bash
python -m unittest discover -s tests -v
```

## Estrutura

```text
app/                 Interface Streamlit e features partilhadas
data/steam.csv       Dataset de demonstração versionado
scripts/train.py     Treino, busca e avaliação
tests/               Testes de features e entradas do modelo
models/               Modelo treinado (gerado localmente)
```
