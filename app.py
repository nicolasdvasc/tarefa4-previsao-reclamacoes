"""
=================================================================================
DASHBOARD INTERATIVO - PREVISÃO DE RECLAMAÇÕES DE CLIENTES
Universidade de Brasília - UnB
Professor: João Gabriel de Moraes Souza

Aluno: Nícolas Duarte Vasconcellos
ID: 200042343
=================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# Métricas
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, auc,
                             precision_score, recall_score, f1_score, accuracy_score)

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Análise de Reclamações - UnB",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado - minimalista, apenas para melhorias visuais básicas
st.markdown("""
    <style>
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1e3a8a;
        font-weight: bold;
    }
    h2 {
        color: #2563eb;
    }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

@st.cache_data
def load_data(uploaded_file=None):
    """Carrega e faz pré-processamento básico dos dados"""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep='\t')
        except:
            df = pd.read_csv(uploaded_file)
    else:
        # URL de exemplo (substitua pela URL real do dataset)
        st.warning("⚠️ Nenhum arquivo enviado. Usando dataset de exemplo.")
        return None
    
    return df

def preprocess_data(df):
    """Pré-processamento completo dos dados"""
    df_clean = df.copy()
    
    # Tratar valores ausentes
    if 'Income' in df_clean.columns:
        df_clean['Income'].fillna(df_clean['Income'].median(), inplace=True)
    
    # Criar features
    if 'Year_Birth' in df_clean.columns:
        df_clean['Age'] = 2025 - df_clean['Year_Birth']
    
    # Total de gastos
    spending_cols = [col for col in df_clean.columns if 'Mnt' in col]
    if spending_cols:
        df_clean['Total_Spending'] = df_clean[spending_cols].sum(axis=1)
    
    # Total de compras
    purchase_cols = [col for col in df_clean.columns if 'Num' in col and 'Purchases' in col]
    if purchase_cols:
        df_clean['Total_Purchases'] = df_clean[purchase_cols].sum(axis=1)
    
    # Total de filhos
    if 'Kidhome' in df_clean.columns and 'Teenhome' in df_clean.columns:
        df_clean['Total_Children'] = df_clean['Kidhome'] + df_clean['Teenhome']
    
    # Remover colunas desnecessárias
    cols_to_drop = ['ID', 'Z_CostContact', 'Z_Revenue', 'Dt_Customer', 'Year_Birth']
    cols_to_drop = [col for col in cols_to_drop if col in df_clean.columns]
    df_clean.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Encoding de variáveis categóricas
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in df_clean.columns and df_clean[col].nunique() < 20:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    
    # Remover outliers extremos (3 IQR)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'Complain']
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    
    return df_clean

def train_model(X_train, y_train, X_test, y_test, model_name, model):
    """Treina e avalia um modelo"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    
    return model, metrics

# =============================================================================
# SIDEBAR - CONTROLES
# =============================================================================

st.sidebar.image("https://www.unb.br/images/logo_unb.png", width=200)
st.sidebar.markdown("---")

st.sidebar.title("⚙️ Configurações")

# Upload do arquivo
uploaded_file = st.sidebar.file_uploader(
    "📁 Carregar Dataset (CSV/TSV)",
    type=['csv', 'tsv', 'txt'],
    help="Faça upload do arquivo 'marketing_campaign.csv'"
)

st.sidebar.markdown("---")

# Configurações de modelagem
st.sidebar.subheader("🔧 Parâmetros de Modelagem")

apply_smote = st.sidebar.checkbox("Aplicar SMOTE", value=True, 
                                   help="Balancear classes usando SMOTE")

use_rfe = st.sidebar.checkbox("Usar RFE para Seleção", value=True,
                               help="Recursive Feature Elimination")

test_size = st.sidebar.slider("Tamanho do Teste (%)", 10, 40, 20, 5) / 100

st.sidebar.markdown("---")

# Seleção de modelos
st.sidebar.subheader("🤖 Modelos a Treinar")

models_to_train = st.sidebar.multiselect(
    "Selecione os modelos:",
    ['KNN', 'SVM', 'Decision Tree', 'Random Forest', 
     'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'MLP'],
    default=['Random Forest', 'XGBoost', 'LightGBM']
)

st.sidebar.markdown("---")

# Informações do aluno
st.sidebar.info("""
**Desenvolvido por:**  
Nícolas Duarte Vasconcellos  
ID: 200042343

**Disciplina:**  
Modelos Supervisionados

**Professor:**  
João Gabriel de Moraes Souza
""")

# =============================================================================
# HEADER PRINCIPAL
# =============================================================================

col_logo1, col_title, col_logo2 = st.columns([1, 3, 1])

with col_title:
    st.title("📊 Análise Preditiva de Reclamações")
    st.markdown("### Sistema Inteligente para Previsão de Comportamento de Clientes")

st.markdown("---")

# =============================================================================
# CARREGAR E PROCESSAR DADOS
# =============================================================================

if uploaded_file is not None:
    # Carregar dados
    with st.spinner("📥 Carregando dados..."):
        df = load_data(uploaded_file)
    
    if df is not None and 'Complain' in df.columns:
        
        # Mostrar informações básicas
        st.success(f"✅ Dataset carregado com sucesso! {df.shape[0]} registros, {df.shape[1]} variáveis")
        
        # Tabs principais
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Exploração", "🔍 Análise", "🤖 Modelagem", 
            "📈 Resultados", "💡 Insights"
        ])
        
        # =====================================================================
        # TAB 1: EXPLORAÇÃO DOS DADOS
        # =====================================================================
        
        with tab1:
            st.header("📋 Exploração dos Dados")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total de Registros", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total de Variáveis", df.shape[1])
            with col3:
                complain_pct = (df['Complain'].sum() / len(df)) * 100
                st.metric("Taxa de Reclamação", f"{complain_pct:.1f}%")
            with col4:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Valores Ausentes", f"{missing_pct:.1f}%")
            
            st.markdown("---")
            
            # Visualização da distribuição da variável alvo
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribuição da Variável Alvo")
                
                class_counts = df['Complain'].value_counts()
                fig = px.bar(
                    x=['Não Reclamou (0)', 'Reclamou (1)'],
                    y=class_counts.values,
                    color=['Não Reclamou', 'Reclamou'],
                    color_discrete_map={'Não Reclamou': '#2ecc71', 'Reclamou': '#e74c3c'},
                    labels={'x': 'Classe', 'y': 'Quantidade'},
                    title='Contagem de Reclamações'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Proporção de Classes")
                
                fig = px.pie(
                    values=class_counts.values,
                    names=['Não Reclamou (0)', 'Reclamou (1)'],
                    color_discrete_sequence=['#2ecc71', '#e74c3c'],
                    title='Proporção de Reclamações'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Filtros interativos
            st.markdown("---")
            st.subheader("🔍 Filtros de Dados")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_col = st.selectbox("Selecione uma variável para análise:", numeric_cols)
            
            if selected_col:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        df, x=selected_col, color='Complain',
                        marginal='box',
                        title=f'Distribuição de {selected_col} por Classe',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(
                        df, x='Complain', y=selected_col,
                        color='Complain',
                        title=f'Box Plot: {selected_col}',
                        color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Tabela de dados
            st.markdown("---")
            st.subheader("📊 Amostra dos Dados")
            st.dataframe(df.head(100), use_container_width=True)
        
        # =====================================================================
        # TAB 2: ANÁLISE ESTATÍSTICA
        # =====================================================================
        
        with tab2:
            st.header("🔍 Análise Estatística Detalhada")
            
            # Pré-processar dados
            with st.spinner("🔄 Processando dados..."):
                df_clean = preprocess_data(df)
            
            st.success("✅ Pré-processamento concluído!")
            
            # Estatísticas descritivas
            st.subheader("📊 Estatísticas Descritivas")
            st.dataframe(df_clean.describe(), use_container_width=True)
            
            # Matriz de correlação
            st.markdown("---")
            st.subheader("🔗 Matriz de Correlação")
            
            numeric_data = df_clean.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Matriz de Correlação entre Variáveis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top correlações com Complain
            if 'Complain' in corr_matrix.columns:
                st.markdown("---")
                st.subheader("🎯 Correlações com a Variável Alvo")
                
                complain_corr = corr_matrix['Complain'].sort_values(ascending=False)[1:11]
                
                fig = px.bar(
                    x=complain_corr.values,
                    y=complain_corr.index,
                    orientation='h',
                    title='Top 10 Variáveis Correlacionadas com Reclamações',
                    labels={'x': 'Correlação', 'y': 'Variável'},
                    color=complain_corr.values,
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # =====================================================================
        # TAB 3: MODELAGEM
        # =====================================================================
        
        with tab3:
            st.header("🤖 Treinamento de Modelos Preditivos")
            
            if st.button("🚀 Iniciar Treinamento", type="primary"):
                
                # Preparar dados
                with st.spinner("🔄 Preparando dados para modelagem..."):
                    df_model = preprocess_data(df)
                    
                    X = df_model.drop('Complain', axis=1)
                    X = X.select_dtypes(include=[np.number])
                    y = df_model['Complain']
                    
                    # Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # SMOTE
                    if apply_smote:
                        smote = SMOTE(random_state=42)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        st.info(f"✅ SMOTE aplicado. Classes balanceadas: {y_train.value_counts().to_dict()}")
                    
                    # Normalização
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
                    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
                    
                    # RFE
                    if use_rfe:
                        from sklearn.linear_model import LogisticRegression
                        n_features = max(10, X_train_scaled.shape[1] // 2)
                        estimator = LogisticRegression(max_iter=1000, random_state=42)
                        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
                        rfe.fit(X_train_scaled, y_train)
                        selected_features = X_train_scaled.columns[rfe.support_].tolist()
                        
                        X_train_scaled = X_train_scaled[selected_features]
                        X_test_scaled = X_test_scaled[selected_features]
                        
                        st.info(f"✅ RFE aplicado. {len(selected_features)} features selecionadas.")
                
                # Treinar modelos
                st.markdown("---")
                st.subheader("📊 Progresso do Treinamento")
                
                models_dict = {}
                results = {}
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                model_mapping = {
                    'KNN': KNeighborsClassifier(n_neighbors=5),
                    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
                    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
                    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
                    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
                }
                
                for idx, model_name in enumerate(models_to_train):
                    status_text.text(f"Treinando {model_name}...")
                    
                    model = model_mapping[model_name]
                    trained_model, metrics = train_model(
                        X_train_scaled, y_train, X_test_scaled, y_test,
                        model_name, model
                    )
                    
                    results[model_name] = metrics
                    models_dict[model_name] = trained_model
                    
                    progress_bar.progress((idx + 1) / len(models_to_train))
                
                status_text.text("✅ Treinamento concluído!")
                
                # Salvar resultados no session state
                st.session_state['results'] = results
                st.session_state['models'] = models_dict
                st.session_state['X_test'] = X_test_scaled
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = X_train_scaled.columns.tolist()
                
                st.success("🎉 Todos os modelos foram treinados com sucesso!")
        
        # =====================================================================
        # TAB 4: RESULTADOS
        # =====================================================================
        
        with tab4:
            st.header("📈 Resultados da Modelagem")
            
            if 'results' in st.session_state:
                results = st.session_state['results']
                
                # Tabela comparativa
                st.subheader("📊 Comparação de Modelos")
                
                comparison_data = []
                for model_name, metrics in results.items():
                    comparison_data.append({
                        'Modelo': model_name,
                        'Acurácia': metrics['accuracy'],
                        'Precisão': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1'],
                        'AUC-ROC': metrics['auc']
                    })
                
                comparison_df = pd.DataFrame(comparison_data).sort_values('AUC-ROC', ascending=False)
                
                # Destacar melhor modelo
                def highlight_max(s):
                    is_max = s == s.max()
                    return ['background-color: #d4edda' if v else '' for v in is_max]
                
                st.dataframe(
                    comparison_df.style.apply(highlight_max, subset=['AUC-ROC']),
                    use_container_width=True
                )
                
                # Melhor modelo
                best_model_name = comparison_df.iloc[0]['Modelo']
                best_auc = comparison_df.iloc[0]['AUC-ROC']
                
                st.success(f"🏆 **Melhor Modelo:** {best_model_name} (AUC-ROC: {best_auc:.4f})")
                
                # Visualizações
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Comparação de Métricas")
                    fig = px.bar(
                        comparison_df,
                        x='Modelo',
                        y=['Acurácia', 'Precisão', 'Recall', 'F1-Score'],
                        barmode='group',
                        title='Métricas de Desempenho por Modelo'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("AUC-ROC Scores")
                    fig = px.bar(
                        comparison_df.sort_values('AUC-ROC'),
                        x='AUC-ROC',
                        y='Modelo',
                        orientation='h',
                        title='Comparação de AUC-ROC',
                        color='AUC-ROC',
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Curvas ROC
                st.markdown("---")
                st.subheader("📉 Curvas ROC")
                
                fig = go.Figure()
                
                for model_name, metrics in results.items():
                    if 'fpr' in metrics and 'tpr' in metrics:
                        fig.add_trace(go.Scatter(
                            x=metrics['fpr'],
                            y=metrics['tpr'],
                            name=f"{model_name} (AUC={metrics['auc']:.3f})",
                            mode='lines'
                        ))
                
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    name='Baseline',
                    line=dict(dash='dash', color='red')
                ))
                
                fig.update_layout(
                    title='Curvas ROC - Comparação',
                    xaxis_title='Taxa de Falsos Positivos',
                    yaxis_title='Taxa de Verdadeiros Positivos',
                    width=800,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Matriz de confusão do melhor modelo
                st.markdown("---")
                st.subheader(f"🎯 Matriz de Confusão - {best_model_name}")
                
                best_cm = results[best_model_name]['confusion_matrix']
                
                fig = px.imshow(
                    best_cm,
                    text_auto=True,
                    labels=dict(x="Predito", y="Real", color="Contagem"),
                    x=['Não Reclamou', 'Reclamou'],
                    y=['Não Reclamou', 'Reclamou'],
                    color_continuous_scale='Blues'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("⚠️ Execute o treinamento dos modelos primeiro (Tab: Modelagem)")
        
        # =====================================================================
        # TAB 5: INSIGHTS
        # =====================================================================
        
        with tab5:
            st.header("💡 Insights e Recomendações")
            
            if 'results' in st.session_state and 'models' in st.session_state:
                results = st.session_state['results']
                models = st.session_state['models']
                feature_names = st.session_state['feature_names']
                
                # Selecionar melhor modelo
                comparison_data = []
                for model_name, metrics in results.items():
                    comparison_data.append({
                        'Modelo': model_name,
                        'AUC-ROC': metrics['auc']
                    })
                
                best_model_name = max(comparison_data, key=lambda x: x['AUC-ROC'])['Modelo']
                best_model = models[best_model_name]
                
                st.subheader(f"🏆 Análise do Melhor Modelo: {best_model_name}")
                
                # Feature Importance
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(15)
                    
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Variáveis Mais Importantes',
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Exibir tabela
                    st.subheader("📋 Tabela de Importância")
                    st.dataframe(importance_df, use_container_width=True)
                
                # Recomendações Gerenciais
                st.markdown("---")
                st.subheader("🎯 Recomendações Estratégicas")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div style='background-color: #dbeafe; border-left: 4px solid #3b82f6; padding: 20px; border-radius: 8px; margin: 10px 0;'>
                        <h4 style='color: #000000; margin: 0 0 10px 0; font-weight: bold;'>📊 Monitoramento Proativo</h4>
                        <ul style='color: #000000; margin: 5px 0; padding-left: 20px; line-height: 1.8;'>
                            <li style='color: #000000;'>Implementar sistema de alerta para clientes de alto risco</li>
                            <li style='color: #000000;'>Score de risco > 70%: ação imediata</li>
                            <li style='color: #000000;'>Score entre 50-70%: monitoramento próximo</li>
                            <li style='color: #000000;'>Priorizar recursos da equipe de suporte</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style='background-color: #d1fae5; border-left: 4px solid #10b981; padding: 20px; border-radius: 8px; margin: 10px 0;'>
                        <h4 style='color: #000000; margin: 0 0 10px 0; font-weight: bold;'>💡 Personalização</h4>
                        <ul style='color: #000000; margin: 5px 0; padding-left: 20px; line-height: 1.8;'>
                            <li style='color: #000000;'>Segmentar clientes por perfil de risco</li>
                            <li style='color: #000000;'>Campanhas customizadas por segmento</li>
                            <li style='color: #000000;'>Ofertas personalizadas para retenção</li>
                            <li style='color: #000000;'>Atendimento diferenciado</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style='background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; border-radius: 8px; margin: 10px 0;'>
                        <h4 style='color: #000000; margin: 0 0 10px 0; font-weight: bold;'>🔄 Melhoria Contínua</h4>
                        <ul style='color: #000000; margin: 5px 0; padding-left: 20px; line-height: 1.8;'>
                            <li style='color: #000000;'>Retreinar modelo mensalmente</li>
                            <li style='color: #000000;'>Monitorar drift de dados</li>
                            <li style='color: #000000;'>Avaliar impacto das ações</li>
                            <li style='color: #000000;'>Ajustar estratégias baseado em resultados</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div style='background-color: #fee2e2; border-left: 4px solid #ef4444; padding: 20px; border-radius: 8px; margin: 10px 0;'>
                        <h4 style='color: #000000; margin: 0 0 10px 0; font-weight: bold;'>⚠️ Pontos de Atenção</h4>
                        <ul style='color: #000000; margin: 5px 0; padding-left: 20px; line-height: 1.8;'>
                            <li style='color: #000000;'>Investigar causas raiz das reclamações</li>
                            <li style='color: #000000;'>Revisar qualidade em categorias críticas</li>
                            <li style='color: #000000;'>Melhorar experiência do cliente</li>
                            <li style='color: #000000;'>Fortalecer canais de comunicação</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Simulador de Predição
                st.markdown("---")
                st.subheader("🔮 Simulador de Predição")
                
                # Texto introdutório com estilo inline para garantir cor preta
                st.markdown("""
                    <div style='background-color: #ffffff; padding: 10px; border-radius: 5px;'>
                        <p style='color: #000000; font-size: 16px; margin: 0;'>
                            Teste o modelo com dados de exemplo:
                        </p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("")  # Espaçamento
                
                if st.button("🎲 Gerar Predição de Exemplo", type="primary"):
                    # Pegar amostra aleatória
                    X_test = st.session_state['X_test']
                    y_test = st.session_state['y_test']
                    
                    sample_idx = np.random.randint(0, len(X_test))
                    sample = X_test.iloc[sample_idx:sample_idx+1]
                    actual_class = y_test.iloc[sample_idx]
                    
                    # Predição
                    pred_proba = best_model.predict_proba(sample)[0, 1]
                    pred_class = best_model.predict(sample)[0]
                    
                    # Exibir resultados
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Probabilidade de Reclamação",
                            f"{pred_proba*100:.1f}%",
                            delta=f"{'Alto Risco' if pred_proba > 0.7 else 'Médio' if pred_proba > 0.5 else 'Baixo'}"
                        )
                    
                    with col2:
                        st.metric(
                            "Predição",
                            "Vai Reclamar" if pred_class == 1 else "Não Vai Reclamar"
                        )
                    
                    with col3:
                        st.metric(
                            "Classe Real",
                            "Reclamou" if actual_class == 1 else "Não Reclamou",
                            delta="Correto ✓" if pred_class == actual_class else "Incorreto ✗"
                        )
                    
                    # Recomendação com texto PRETO garantido via inline style
                    st.write("")  # Espaçamento
                    
                    if pred_proba > 0.7:
                        st.markdown("""
                        <div style='background-color: #fee2e2; border-left: 4px solid #dc2626; padding: 20px; border-radius: 8px; margin: 10px 0;'>
                            <h4 style='color: #000000; margin: 0 0 10px 0; font-weight: bold;'>🚨 AÇÃO URGENTE RECOMENDADA</h4>
                            <ul style='color: #000000; margin: 5px 0; padding-left: 20px; line-height: 1.8;'>
                                <li style='color: #000000;'>Contato imediato da equipe de retenção</li>
                                <li style='color: #000000;'>Oferecer benefício especial</li>
                                <li style='color: #000000;'>Investigar histórico recente</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif pred_proba > 0.5:
                        st.markdown("""
                        <div style='background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 20px; border-radius: 8px; margin: 10px 0;'>
                            <h4 style='color: #000000; margin: 0 0 10px 0; font-weight: bold;'>⚠️ MONITORAMENTO RECOMENDADO</h4>
                            <ul style='color: #000000; margin: 5px 0; padding-left: 20px; line-height: 1.8;'>
                                <li style='color: #000000;'>Incluir em lista de acompanhamento</li>
                                <li style='color: #000000;'>Enviar pesquisa de satisfação</li>
                                <li style='color: #000000;'>Monitorar próximas interações</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='background-color: #d1fae5; border-left: 4px solid #10b981; padding: 20px; border-radius: 8px; margin: 10px 0;'>
                            <h4 style='color: #000000; margin: 0 0 10px 0; font-weight: bold;'>✅ CLIENTE DE BAIXO RISCO</h4>
                            <ul style='color: #000000; margin: 5px 0; padding-left: 20px; line-height: 1.8;'>
                                <li style='color: #000000;'>Manter atendimento padrão</li>
                                <li style='color: #000000;'>Continuar estratégia atual</li>
                                <li style='color: #000000;'>Acompanhamento de rotina</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
            else:
                st.warning("⚠️ Execute o treinamento dos modelos primeiro (Tab: Modelagem)")
    
    else:
        st.error("❌ Erro ao carregar dataset ou coluna 'Complain' não encontrada!")

else:
    # Página inicial quando não há arquivo carregado
    st.markdown("""
    ## 👋 Bem-vindo ao Sistema de Previsão de Reclamações
    
    Este dashboard interativo foi desenvolvido como parte da **Tarefa 4** da disciplina de 
    **Modelos Supervisionados** da Universidade de Brasília.
    
    ### 📚 Sobre o Projeto
    
    O objetivo é desenvolver um modelo preditivo para identificar clientes com maior 
    probabilidade de reclamação, permitindo:
    
    - 🎯 **Ações proativas** de retenção
    - 💡 **Personalização** do atendimento
    - 📊 **Otimização** de recursos
    - 🔍 **Insights** estratégicos
    
    ### 🚀 Como Usar
    
    1. **Faça upload** do dataset 'marketing_campaign.csv' na barra lateral
    2. **Configure** os parâmetros de modelagem
    3. **Selecione** os modelos que deseja treinar
    4. **Explore** os resultados nas diferentes abas
    
    ### 📁 Dataset Necessário
    
    - **Nome:** Customer Personality Analysis
    - **Fonte:** Kaggle
    - **Formato:** CSV/TSV
    - **Variável Alvo:** Complain (0 = Não reclamou, 1 = Reclamou)
    
    ### 🎓 Informações Acadêmicas
    
    - **Aluno:** Nícolas Duarte Vasconcellos
    - **ID:** 200042343
    - **Professor:** João Gabriel de Moraes Souza
    - **Disciplina:** Engenharia de Produção - Modelos Supervisionados
    
    ---
    
    ### 📊 Funcionalidades do Dashboard
    
    #### 📋 Exploração
    - Visualização interativa dos dados
    - Estatísticas descritivas
    - Filtros dinâmicos
    - Análise de distribuições
    
    #### 🔍 Análise
    - Matriz de correlação
    - Análise estatística avançada
    - Identificação de padrões
    - Visualizações interativas
    
    #### 🤖 Modelagem
    - Seleção dinâmica de variáveis
    - Aplicação de SMOTE para balanceamento
    - RFE para seleção de features
    - Treinamento de múltiplos modelos
    
    #### 📈 Resultados
    - Comparação de métricas
    - Curvas ROC
    - Matrizes de confusão
    - Rankings de performance
    
    #### 💡 Insights
    - Importância de variáveis
    - Recomendações estratégicas
    - Simulador de predições
    - Interpretação gerencial
    
    ---
    
    ### ⚙️ Modelos Disponíveis
    
    **Baseados em Distância:**
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    
    **Bagging:**
    - Decision Tree
    - Random Forest
    
    **Boosting:**
    - AdaBoost
    - Gradient Boosting
    - XGBoost
    - LightGBM
    
    **Redes Neurais:**
    - Multi-Layer Perceptron (MLP)
    
    ---
    
    ### 📞 Suporte
    
    Em caso de dúvidas ou problemas, consulte a documentação do projeto ou 
    entre em contato através dos canais oficiais da disciplina.
    
    """)
    
    # Imagem ilustrativa (opcional)
    st.image("https://via.placeholder.com/800x400.png?text=Sistema+de+Previsão+de+Reclamações", 
             use_container_width=True)
    
    # Botão de exemplo
    st.info("👈 **Comece fazendo upload do dataset na barra lateral!**")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Dashboard de Análise Preditiva</strong></p>
    <p>Universidade de Brasília - Departamento de Engenharia de Produção</p>
    <p>Desenvolvido por Nícolas Duarte Vasconcellos (200042343)</p>
    <p>© 2025 - Todos os direitos reservados</p>
</div>
""", unsafe_allow_html=True)
