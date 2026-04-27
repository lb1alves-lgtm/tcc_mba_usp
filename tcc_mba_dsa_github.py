import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
import warnings
import sys
import os

warnings.filterwarnings("ignore")

# =========================================================
# 1. MAPEAMENTO DAS EQUAÇÕES E CONFIGURAÇÕES
# =========================================================
equacoes = {
    'cfu_rh': ['ger_ee', 'pm_ee', 'prec_plu'],
    'cf_em': ['pro_min', 'ipi_comm', 'tx_camb'],
    'cp_roy_ita': ['ger_ee', 'prec_plu', 'pm_ee'],
    'cf_ep': ['pro_petr', 'p_petr_brent', 'tx_camb'],
    'dcl': ['dcl_lag1', 'rcl', 'tx_selic', 'rp']
}

HORIZONTE_MESES = 360 # 2055
SPLITS_CV = 3 
DIRETORIO_SAIDA = r'C:\Users\luiz.alves\Desktop\MBA_DSA_USP_GitHub'

# Variáveis para cálculo do Valor Presente
TAXA_DESCONTO = 0.1475
ANO_REFERENCIA_VP = 2025

def create_model_pipeline(model):
    return Pipeline([('model', model)])

param_grid_rf = {
    'model__n_estimators': stats.randint(50, 300),
    'model__max_depth': stats.randint(3, 15),
    'model__min_samples_split': stats.randint(2, 10)
}

param_grid_xgb = {
    'model__n_estimators': stats.randint(50, 300),
    'model__max_depth': stats.randint(3, 9),
    'model__learning_rate': stats.uniform(0.01, 0.19), 
    'model__subsample': stats.uniform(0.8, 0.2)        
}

# =========================================================
# 2. CARREGAMENTO E TRATAMENTO DA BASE DE DADOS
# =========================================================
nome_arquivo = os.path.join(DIRETORIO_SAIDA, 'dados_gerais_github.xlsx')

try:
    df = pd.read_excel(nome_arquivo) 
except FileNotFoundError:
    sys.exit(f"\nERRO: Ficheiro não encontrado no diretório: {nome_arquivo}")

df.columns = df.columns.str.strip()

dicionario_renomeio = {
    'ano_mes': 'data',
    'Contr_Fundo_Utiliz_Rec_Hidr (R$) - deflacionado': 'cfu_rh',
    'Contr_Fundo_Extr_Min (R$) - Deflacionado': 'cf_em',
    'Contr_Partic_Royalt_Itaipu (R$) - Deflacionado': 'cp_roy_ita',
    'Contr_Fundo_Extr_Petr (R$) - Deflacionado': 'cf_ep',
    'Produção mineral (R$) - Deflacionado': 'pro_min',
    'Índice de Preço internacional das commodities': 'ipi_comm',
    'Taxas de câmbio (R$/US$)': 'tx_camb',
    'Geração de energia elétrica - (MWh)': 'ger_ee',
    'Preço médio da energia elétrica (R$) - Deflacionado': 'pm_ee',
    'Precipitação Pluviométrica (mm)': 'prec_plu',
    'Volume de produção de petróleo e gás (barris)': 'pro_petr',
    'Preço internacional do petróleo Brent (US$/barril)': 'p_petr_brent',
    'Dívida Consolidada Líquida (R$) - Deflacionado': 'dcl',
    'Resultado Primário (R$) - Deflacionado': 'rp',
    'Taxa de Juros Selic (%)': 'tx_selic',
    'Receita Corrente Líquida (R$) - Deflacionado': 'rcl'
}

df = df.rename(columns=dicionario_renomeio)

try:
    df['data'] = pd.to_datetime(df['data'].astype(str), format='%Y%m')
    df.set_index('data', inplace=True)
    df = df.asfreq('MS')
except KeyError:
    sys.exit("ERRO: Verifique se a coluna de tempo se chama realmente 'ano_mes'.")

df = df.ffill().bfill()

if 'dcl' in df.columns:
    df['dcl_lag1'] = df['dcl'].shift(1)

df = df.iloc[1:]

# =========================================================
# 3. FUNÇÕES AUXILIARES
# =========================================================
def testes_estacionaridade_completos(serie, nome):
    serie_limpa = serie.dropna()
    adf_res = adfuller(serie_limpa)
    kpss_res = kpss(serie_limpa, regression='c', nlags='auto')
    
    return {
        'Variavel': nome, 
        'ADF_P_Valor': adf_res[1], 
        'ADF_Indica_Estacionaria?': 'Sim' if adf_res[1] <= 0.05 else 'Não',
        'KPSS_P_Valor': kpss_res[1],
        'KPSS_Indica_Estacionaria?': 'Sim' if kpss_res[1] >= 0.05 else 'Não'
    }

def projetar_exogenas(df_historico, colunas_exogenas, periodos):
    df_futuro = pd.DataFrame(index=pd.date_range(start='2025-01-01', periods=periodos, freq='MS'))
    for col in colunas_exogenas:
        if col == 'dcl_lag1': continue 
        df_prophet = pd.DataFrame({'ds': df_historico.index, 'y': df_historico[col].values})
        m = Prophet(seasonality_mode='multiplicative')
        m.fit(df_prophet)
        future = m.make_future_dataframe(periods=periodos, freq='MS')
        forecast = m.predict(future)
        df_futuro[col] = forecast.iloc[-periodos:]['yhat'].values
    return df_futuro

def calc_metricas(y_true, y_pred):
    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred)
    }

# =========================================================
# 4. LOOP PRINCIPAL
# =========================================================
lista_testes = []
lista_vencedores = []
lista_diagnosticos = []
lista_todas_metricas = [] 
df_projecoes_mensais = pd.DataFrame(index=pd.date_range(start='2025-01-01', periods=HORIZONTE_MESES, freq='MS'))

# String para armazenar todos os relatórios MQO em texto
relatorio_mqo_texto = "SUMÁRIO DE REGRESSÕES MQO (OLS)\n"

tscv = TimeSeriesSplit(n_splits=SPLITS_CV, test_size=12)

for target, exog_cols in equacoes.items():
    print(f"\n{'='*50}\nIniciando Modelagem Robusta para: {target.upper()}\n{'='*50}")
        
    res_teste = testes_estacionaridade_completos(df[target], target)
    lista_testes.append(res_teste)

    try:
        dec = seasonal_decompose(df[target].dropna(), model='additive', period=12)
        fig = dec.plot()
        fig.set_size_inches(10, 8)
        plt.suptitle(f'Decomposição Clássica - {target.upper()}', y=1.02)
        plt.savefig(os.path.join(DIRETORIO_SAIDA, f'Decomposicao_{target}.png'), bbox_inches='tight')
        plt.close(fig)
    except Exception:
        pass

    modelos_cv = {
        'SARIMA (Box-Jenkins)': {'rmse': [], 'mae': [], 'mape': []},
        'Prophet': {'rmse': [], 'mae': [], 'mape': []},
        'Random_Forest (Pipeline)': {'rmse': [], 'mae': [], 'mape': []},
        'XGBoost (Pipeline)': {'rmse': [], 'mae': [], 'mape': []}
    }

    for fold, (train_index, test_index) in enumerate(tscv.split(df)):
        print(f"  -> A processar Janela de Validação {fold + 1}/{SPLITS_CV}...")
        
        train, test = df.iloc[train_index], df.iloc[test_index]
        y_train, y_test = train[target], test[target]
        X_train, X_test = train[exog_cols], test[exog_cols]

        # SARIMA
        try:
            import pmdarima as pm
            modelo_arima = pm.auto_arima(y_train, X=X_train, seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action="ignore")
            pred_sarima = modelo_arima.predict(n_periods=len(y_test), X=X_test)
            m_sarima = calc_metricas(y_test, pred_sarima)
            for k in m_sarima: modelos_cv['SARIMA (Box-Jenkins)'][k].append(m_sarima[k])
        except: pass

        # Prophet
        df_pro_train = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})
        for col in exog_cols: df_pro_train[col] = X_train[col].values
        prophet_m = Prophet()
        for col in exog_cols: prophet_m.add_regressor(col)
        prophet_m.fit(df_pro_train)
        df_pro_test = pd.DataFrame({'ds': y_test.index})
        for col in exog_cols: df_pro_test[col] = X_test[col].values
        pred_prophet = prophet_m.predict(df_pro_test)['yhat'].values
        m_prophet = calc_metricas(y_test, pred_prophet)
        for k in m_prophet: modelos_cv['Prophet'][k].append(m_prophet[k])

        tscv_inner = TimeSeriesSplit(n_splits=2) 

        # Random Forest
        rf_pipeline = create_model_pipeline(RandomForestRegressor(random_state=42))
        rf_search = RandomizedSearchCV(rf_pipeline, param_grid_rf, n_iter=10, cv=tscv_inner, scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42)
        rf_search.fit(X_train, y_train)
        pred_rf = rf_search.best_estimator_.predict(X_test)
        m_rf = calc_metricas(y_test, pred_rf)
        for k in m_rf: modelos_cv['Random_Forest (Pipeline)'][k].append(m_rf[k])

        # XGBoost
        xgb_pipeline = create_model_pipeline(XGBRegressor(random_state=42, objective='reg:squarederror'))
        xgb_search = RandomizedSearchCV(xgb_pipeline, param_grid_xgb, n_iter=10, cv=tscv_inner, scoring='neg_root_mean_squared_error', n_jobs=-1, random_state=42)
        xgb_search.fit(X_train, y_train)
        pred_xgb = xgb_search.best_estimator_.predict(X_test)
        m_xgb = calc_metricas(y_test, pred_xgb)
        for k in m_xgb: modelos_cv['XGBoost (Pipeline)'][k].append(m_xgb[k])

    # Consolidação e escolha do Vencedor
    melhor_nome = None
    menor_rmse = float('inf')
    
    for modelo_nome, metricas in modelos_cv.items():
        if len(metricas['rmse']) > 0:
            rmse_medio = np.mean(metricas['rmse'])
            mae_medio = np.mean(metricas['mae'])
            mape_medio = np.mean(metricas['mape'])
            
            lista_todas_metricas.append({
                'Variavel': target,
                'Modelo': modelo_nome,
                'RMSE_Medio': rmse_medio,
                'MAE_Medio': mae_medio,
                'MAPE_Medio': mape_medio
            })
            
            if rmse_medio < menor_rmse:
                menor_rmse = rmse_medio
                melhor_nome = modelo_nome
    
    print(f"\n Melhor modelo: {target.upper()}: {melhor_nome} (RMSE: {menor_rmse:.4f})")
    lista_vencedores.append({'Variavel': target, 'Modelo_Vencedor': melhor_nome, 'RMSE_Medio_CV': menor_rmse})

    # =========================================================
    # 5. TREINAMENTO FINAL E PROJEÇÃO
    # =========================================================
    y_full = df[target]
    X_full = df[exog_cols]

    # --- INÍCIO: GERAÇÃO DO RELATÓRIO MQO EM TXT E PNG ---
    try:
        X_full_const = sm.add_constant(X_full) 
        modelo_mqo = sm.OLS(y_full, X_full_const).fit()
        resumo_texto = modelo_mqo.summary().as_text()
        
        # 1. Adiciona ao texto geral (TXT)
        relatorio_mqo_texto += f"\n\n{'='*60}\n"
        relatorio_mqo_texto += f"VARIÁVEL DEPENDENTE: {target.upper()}\n"
        relatorio_mqo_texto += f"VARIÁVEIS INDEPENDENTES: {', '.join(exog_cols)}\n"
        relatorio_mqo_texto += f"{'='*60}\n"
        relatorio_mqo_texto += resumo_texto
        
        # 2. Gera e salva a imagem PNG individual para a variável atual
        fig_mqo, ax_mqo = plt.subplots(figsize=(10, 7))
        ax_mqo.axis('off')
        ax_mqo.text(0.01, 0.99, resumo_texto, fontfamily='monospace', fontsize=10, va='top', ha='left')
        
        caminho_png_mqo = os.path.join(DIRETORIO_SAIDA, f'Sumario_MQO_{target}.png')
        plt.savefig(caminho_png_mqo, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close(fig_mqo)
        print(f"  Tabela MQO salva como imagem: Sumario_MQO_{target}.png")
        
    except Exception as e:
        print(f"Erro ao gerar relatório MQO para {target}: {e}")
    # --- FIM: GERAÇÃO DO RELATÓRIO MQO ---
    
    if melhor_nome == 'SARIMA (Box-Jenkins)':
        modelo_final = pm.auto_arima(y_full, X=X_full, seasonal=True, m=12, stepwise=True, suppress_warnings=True)
        residuos = modelo_final.resid()
        
    elif melhor_nome == 'Prophet':
        df_pro_full = pd.DataFrame({'ds': y_full.index, 'y': y_full.values})
        for col in exog_cols: df_pro_full[col] = X_full[col].values
        
        modelo_final_95 = Prophet(interval_width=0.95)
        for col in exog_cols: modelo_final_95.add_regressor(col)
        modelo_final_95.fit(df_pro_full)
        
        modelo_final_90 = Prophet(interval_width=0.90)
        for col in exog_cols: modelo_final_90.add_regressor(col)
        modelo_final_90.fit(df_pro_full)
        
        residuos = y_full.values - modelo_final_95.predict(df_pro_full)['yhat'].values
        
    elif melhor_nome == 'Random_Forest (Pipeline)':
        modelo_final = rf_search.best_estimator_.fit(X_full, y_full)
        residuos = y_full.values - modelo_final.predict(X_full)
        
    else: 
        modelo_final = xgb_search.best_estimator_.fit(X_full, y_full)
        residuos = y_full.values - modelo_final.predict(X_full)

    # --- TESTES DE DIAGNÓSTICO DOS RESÍDUOS ---
    jb_stat, jb_p = stats.jarque_bera(residuos)
    lb_df = acorr_ljungbox(residuos, lags=[12], return_df=True)
    lb_p = lb_df['lb_pvalue'].iloc[0]
    
    lista_diagnosticos.append({
        'Variavel': target,
        'Modelo_Vencedor': melhor_nome,
        'Jarque_Bera_P_Valor': jb_p,
        'Residuos_Normais?': 'Sim' if jb_p > 0.05 else 'Não',
        'Ljung_Box_P_Valor': lb_p,
        'E_Ruido_Branco? (Sem Autocorrelacao)': 'Sim' if lb_p > 0.05 else 'Não'
    })

    # --- PROJEÇÃO DO FUTURO ---
    X_futuro = projetar_exogenas(df, exog_cols, HORIZONTE_MESES)
    
    previsoes_alvo = []
    lim_inf_95 = []
    lim_sup_95 = []
    lim_inf_90 = []
    lim_sup_90 = []
    
    std_resid = np.std(residuos) 

    if target == 'dcl':
        ultimo_dcl = df['dcl'].iloc[-1]
        X_futuro_mod = X_futuro.copy()
        for i in range(HORIZONTE_MESES):
            X_futuro_mod.loc[X_futuro_mod.index[i], 'dcl_lag1'] = ultimo_dcl
            x_step = pd.DataFrame(X_futuro_mod.iloc[i]).T
            x_step = x_step[exog_cols]
            
            if melhor_nome == 'SARIMA (Box-Jenkins)': 
                pred_step, conf_95 = modelo_final.predict(n_periods=1, X=x_step, return_conf_int=True, alpha=0.05)
                _, conf_90 = modelo_final.predict(n_periods=1, X=x_step, return_conf_int=True, alpha=0.10)
                
                pred_val = pred_step.values[0] if isinstance(pred_step, pd.Series) else pred_step[0]
                previsoes_alvo.append(pred_val)
                lim_inf_95.append(conf_95[0][0])
                lim_sup_95.append(conf_95[0][1])
                lim_inf_90.append(conf_90[0][0])
                lim_sup_90.append(conf_90[0][1])
                ultimo_dcl = pred_val
                
            elif melhor_nome == 'Prophet':
                df_pro_step = pd.DataFrame({'ds': [X_futuro_mod.index[i]]})
                for col in exog_cols: df_pro_step[col] = x_step[col].values
                
                forecast_95 = modelo_final_95.predict(df_pro_step)
                forecast_90 = modelo_final_90.predict(df_pro_step)
                
                pred_val = forecast_95['yhat'].values[0]
                previsoes_alvo.append(pred_val)
                lim_inf_95.append(forecast_95['yhat_lower'].values[0])
                lim_sup_95.append(forecast_95['yhat_upper'].values[0])
                lim_inf_90.append(forecast_90['yhat_lower'].values[0])
                lim_sup_90.append(forecast_90['yhat_upper'].values[0])
                ultimo_dcl = pred_val
                
            else: 
                pred_val = modelo_final.predict(x_step)[0]
                previsoes_alvo.append(pred_val)
                lim_inf_95.append(pred_val - 1.96 * std_resid)
                lim_sup_95.append(pred_val + 1.96 * std_resid)
                lim_inf_90.append(pred_val - 1.645 * std_resid)
                lim_sup_90.append(pred_val + 1.645 * std_resid)
                ultimo_dcl = pred_val
                
    else:
        X_futuro = X_futuro[exog_cols]
        if melhor_nome == 'SARIMA (Box-Jenkins)': 
            pred_array, conf_95 = modelo_final.predict(n_periods=HORIZONTE_MESES, X=X_futuro, return_conf_int=True, alpha=0.05)
            _, conf_90 = modelo_final.predict(n_periods=HORIZONTE_MESES, X=X_futuro, return_conf_int=True, alpha=0.10)
            
            previsoes_alvo = np.array(pred_array)
            lim_inf_95 = conf_95[:, 0]
            lim_sup_95 = conf_95[:, 1]
            lim_inf_90 = conf_90[:, 0]
            lim_sup_90 = conf_90[:, 1]
            
        elif melhor_nome == 'Prophet':
            df_pro_fut = pd.DataFrame({'ds': X_futuro.index})
            for col in exog_cols: df_pro_fut[col] = X_futuro[col].values
            
            forecast_95 = modelo_final_95.predict(df_pro_fut)
            forecast_90 = modelo_final_90.predict(df_pro_fut)
            
            previsoes_alvo = forecast_95['yhat'].values
            lim_inf_95 = forecast_95['yhat_lower'].values
            lim_sup_95 = forecast_95['yhat_upper'].values
            lim_inf_90 = forecast_90['yhat_lower'].values
            lim_sup_90 = forecast_90['yhat_upper'].values
            
        else: 
            previsoes_alvo = modelo_final.predict(X_futuro)
            lim_inf_95 = previsoes_alvo - 1.96 * std_resid
            lim_sup_95 = previsoes_alvo + 1.96 * std_resid
            lim_inf_90 = previsoes_alvo - 1.645 * std_resid
            lim_sup_90 = previsoes_alvo + 1.645 * std_resid

    # Salvando APENAS a Projeção Principal no DataFrame Mensal
    df_projecoes_mensais[f"{target} ({melhor_nome})"] = previsoes_alvo

    # --- PLOTAGEM DO GRÁFICO ---
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, y_full, label='Histórico Real', color='#1f77b4', linewidth=2)
    plt.plot(X_futuro.index, previsoes_alvo, label=f'Projeção ({melhor_nome})', color='#ff7f0e', linewidth=2)
    
    plt.fill_between(X_futuro.index, lim_inf_95, lim_sup_95, color='#ff7f0e', alpha=0.15, label='IC 95%')
    plt.fill_between(X_futuro.index, lim_inf_90, lim_sup_90, color='#ff7f0e', alpha=0.35, label='IC 90%')
    
    plt.title(f'Projeção Mensal com Múltiplos ICs (90% e 95%): {target.upper()}', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Valor')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    caminho_grafico_proj = os.path.join(DIRETORIO_SAIDA, f'Projecao_IC_Duplo_{target}.png')
    plt.savefig(caminho_grafico_proj, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Gráfico com IC 90% e 95% salvo para {target.upper()}.")

# =========================================================
# 6. EXPORTANDO PARA EXCEL, TXT E FINALIZAÇÃO
# =========================================================
# 6.1 Agrupa em anos nominais somando os 12 meses
df_projecoes_anuais = df_projecoes_mensais.groupby(df_projecoes_mensais.index.year).sum()
df_projecoes_anuais.index.name = 'Ano'

# 6.2 Calcula o Valor Presente dos totais anuais
df_projecoes_vp = df_projecoes_anuais.copy()
n_anos = df_projecoes_vp.index - ANO_REFERENCIA_VP
fator_desconto = (1 + TAXA_DESCONTO) ** n_anos
df_projecoes_vp = df_projecoes_vp.divide(fator_desconto, axis=0) 

# 6.3 Calcula a soma Acumulada do Valor Presente anualmente
df_projecoes_vp_acumulado = df_projecoes_vp.cumsum()

# 6.4 Consolida os DataFrames de Relatório
df_testes = pd.DataFrame(lista_testes)
df_vencedores = pd.DataFrame(lista_vencedores)
df_diagnosticos = pd.DataFrame(lista_diagnosticos)
df_metricas_todas = pd.DataFrame(lista_todas_metricas) 

caminho_saida = os.path.join(DIRETORIO_SAIDA, 'Resultados_Finais_Projecao_30_Anos.xlsx')

with pd.ExcelWriter(caminho_saida, engine='xlsxwriter') as writer:
    df_projecoes_mensais.to_excel(writer, sheet_name='Projecoes_Mensais')
    df_projecoes_anuais.to_excel(writer, sheet_name='Projecoes_Anuais_Soma')
    df_projecoes_vp.to_excel(writer, sheet_name='Projecoes_VP_Anuais') 
    df_projecoes_vp_acumulado.to_excel(writer, sheet_name='Projecoes_VP_Acumulado') 
    
    df_vencedores.to_excel(writer, sheet_name='Modelos_Vencedores', index=False)
    df_metricas_todas.to_excel(writer, sheet_name='Todas_Metricas', index=False) 
    df_testes.to_excel(writer, sheet_name='Testes_Estatisticos', index=False)
    df_diagnosticos.to_excel(writer, sheet_name='Diagnostico_Residuos', index=False) 

# Exportar o arquivo TXT com o compilado de regressões MQO
caminho_txt_mqo = os.path.join(DIRETORIO_SAIDA, 'Sumario_Regressoes_MQO.txt')
with open(caminho_txt_mqo, 'w', encoding='utf-8') as f:
    f.write(relatorio_mqo_texto)

print(f"\nProcessamento concluído com sucesso!")
print(f"Ficheiro Excel salvo em: '{caminho_saida}'")
print(f"Relatório de Regressões MQO (TXT) salvo em: '{caminho_txt_mqo}'")
