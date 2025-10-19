# 🎮 Steam Price Predictor

Um projeto de **Ciência de Dados + Streamlit** que prevê o preço provável de um jogo na Steam com base em estatísticas públicas: número de donos, avaliações positivas/negativas, tempo médio jogado, ano de lançamento e gênero.

---

## 🚀 Visão Geral

O projeto aplica **aprendizado de máquina supervisionado** (Random Forest Regressor) sobre mais de **24 mil jogos** coletados da Steam.  
A aplicação web (via **Streamlit**) permite prever o preço de um jogo e explorar visualmente os dados.

---

## 📁 Estrutura do Projeto

```
steam_price_predictor/
│
├── app/
│   ├── app.py                 # Interface Streamlit
│   └── steam_price_model.pkl  # Modelo local (opcional)
│
├── data/
│   └── steam.csv              # Dataset principal
│
├── models/
│   └── final_steam_model.pkl  # Modelo final salvo
│
├── scripts/
│   └── train.py               # Script de treinamento
│
├── notebooks/
│   └── EDA.ipynb              # Análise exploratória opcional
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 Tecnologias Principais

- **Python 3.13**
- **Pandas / NumPy**
- **Scikit-learn**
- **Plotly**
- **Streamlit**
- **Joblib**

---

## ⚙️ Treinamento do Modelo

```bash
cd scripts
python train.py
```

O script:
1. Faz *feature engineering* (transforma colunas brutas em variáveis úteis).
2. Executa `RandomizedSearchCV` com 5-fold cross-validation.
3. Avalia com R² e RMSE.
4. Salva o pipeline completo em `/models/final_steam_model.pkl`.

---

## 🌐 Executando o App

```bash
streamlit run app/app.py
```

Abra o link exibido no terminal (`http://localhost:8501`).

**No Streamlit Cloud:**  
garanta que os arquivos estejam em `/app` e `/models`, e o dataset no repositório (para demonstração local).

---

## 📊 Recursos da Aplicação

- **Predição de preço** com base em entradas personalizáveis.
- **Visualização das features mais relevantes.**
- **Comparativo de múltiplos cenários.**
- **Exploração interativa de datasets carregados.**

---

## 📈 Métricas de Desempenho

- **Modelo:** RandomForestRegressor  
- **R²:** ≈ 0.75  
- **RMSE:** ≈ 4.5  
- **Dataset:** Steam Games (SteamSpy)

---

## 🧩 Melhorias Futuras

- Normalização automática de novas colunas (`support for DLC, VR`).
- Deploy contínuo no **Streamlit Cloud** com CI/CD.
- API REST (FastAPI) para integração externa.
- Visual analytics mais detalhado com `plotly.subplots`.

---

## 🧑‍💻 Autor

Desenvolvido por **Zara Takion** — estudante de Sistemas para Internet e entusiasta de dados e web design.
