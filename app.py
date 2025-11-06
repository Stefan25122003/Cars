import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# --- 1. CONFIGURARE APLICATIE ---
st.set_page_config(layout="wide", page_title="Analiză Autoturisme Simplificată")

st.title("🚗 Analiza Factorilor de Preț Auto (Set Simplificat)")
st.markdown("Aplicație interactivă pentru explorarea datelor auto și predicția prețului pe baza coloanelor disponibile (model, year, price, transmission, mileage, fuelType, tax, mpg, engineSize, Manufacturer).")

# --- 1. ÎNCĂRCAREA ȘI PREGĂTIREA DATELOR ---
@st.cache_data
def load_and_prep_data():
    try:
        df = pd.read_csv('CarsData.csv')
        
        df = df.rename(columns={
            'year': 'Year', 
            'price': 'Price', 
            'transmission': 'Gear_Box_Type', 
            'mileage': 'Mileage_Num', 
            'fuelType': 'Fuel_Type', 
            'engineSize': 'Engine_Volume',
            'Manufacturer': 'Manufacturer'
        })
        CONVERSION_RATE = 1.60934 

        # Aplică conversia. Acum, 'Mileage_Num' va fi în Kilometri.
        df['Mileage_Num'] = df['Mileage_Num'] * CONVERSION_RATE
        
        df['Mileage_Num'] = pd.to_numeric(df['Mileage_Num'], errors='coerce')
        df['Engine_Volume'] = pd.to_numeric(df['Engine_Volume'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['tax'] = pd.to_numeric(df['tax'], errors='coerce')
        df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')

        required_cols = ['Price', 'Mileage_Num', 'Engine_Volume', 'mpg', 'tax', 'model']
        df.dropna(subset=required_cols, inplace=True)
        
        df['Year'] = df['Year'].astype(int)
        
        return df

    except Exception as e:
        st.error(f"Eroare la încărcarea sau pregătirea datelor. Asigurați-vă că fișierul `CarsData.csv` există și are structura corectă: {e}")
        st.stop()
        
df = load_and_prep_data()

numeric_features = ['Price', 'Mileage_Num', 'Engine_Volume', 'Year', 'tax', 'mpg']

# --- CALCUL SCOR CALITATE ---
# --- CALCUL SCOR CALITATE ---
try:
    # --- Componentele Scorului (0-1) ---
    df['MPG_Score'] = (df['mpg'] - df['mpg'].min()) / (df['mpg'].max() - df['mpg'].min())
    
    min_eng = df['Engine_Volume'].min()
    max_eng = df['Engine_Volume'].max()
    df['Engine_Score'] = 1 - (df['Engine_Volume'] - min_eng) / (max_eng - min_eng)

    min_year = df['Year'].min()
    max_year = df['Year'].max()
    df['Year_Score'] = (df['Year'] - min_year) / (max_year - min_year)
    
    min_mileage = df['Mileage_Num'].min()
    max_mileage = df['Mileage_Num'].max()
    df['Mileage_Score'] = 1 - (df['Mileage_Num'] - min_mileage) / (max_mileage - min_mileage)
    
    # ******************* MODIFICARE 1: SCOR CALITATE (0-10) *******************
    # Ponderile însumează acum 10 (3+3+2+2)
    df['Quality_Score'] = (
        (df['Year_Score'] * 3) + 
        (df['MPG_Score'] * 3) + 
        (df['Engine_Score'] * 2) + 
        (df['Mileage_Score'] * 2)
    )
    # Rezultatul este direct pe o scară de la 0 la 10.
    
    # ******************* MODIFICARE 2: RAPORT PREȚ-CALITATE (0-10) *******************
    
    # 1. Calculăm raportul brut "value-for-money" (Calitate / Preț)
    # Folosim df['Price'] + 1 pentru a evita împărțirea la zero dacă un preț este 0
    df['Raw_Ratio'] = df['Quality_Score'] / ((df['Price'] / 1000.0) + 1)
    
    # Tratăm valorile anormale care ar putea apărea
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Raw_Ratio'].fillna(0, inplace=True)

    # 2. Normalizăm acest raport brut pe o scară de la 0 la 10
    min_raw_ratio = df['Raw_Ratio'].min()
    max_raw_ratio = df['Raw_Ratio'].max()
    
    if min_raw_ratio == max_raw_ratio:
         df['Price_Quality_Ratio'] = 5.0 # Daca toate valorile sunt egale, dam un scor mediu
    else:
         # Formula Min-Max Scaling ajustată pentru 0-10
         df['Price_Quality_Ratio'] = 10 * (df['Raw_Ratio'] - min_raw_ratio) / (max_raw_ratio - min_raw_ratio)

    # ***********************************************************************************

    # Agregarea datelor (rămâne neschimbată, acum folosește noile scoruri)
    ranking_df = df.groupby(['Manufacturer', 'model']).agg(
        Pret_Mediu = ('Price', 'median'),
        Scor_Calitate_Mediu = ('Quality_Score', 'median'),
        Raport_Pret_Calitate_Mediu = ('Price_Quality_Ratio', 'median'),
        Numar_Inregistrari = ('Year', 'count'),
        # Componente pentru Radar Chart (acestea sunt tot 0-1, ceea ce e corect)
        Scor_An_Mediu = ('Year_Score', 'median'),
        Scor_Consum_Mediu = ('MPG_Score', 'median'),
        Scor_Motor_Mediu = ('Engine_Score', 'median'),
        Scor_Rulaj_Mediu = ('Mileage_Score', 'median')
    ).reset_index()
    
    ranking_df = ranking_df[ranking_df['Numar_Inregistrari'] >= 2]
    ranking_df = ranking_df.sort_values(by='Raport_Pret_Calitate_Mediu', ascending=False)
    
except Exception as e:
    st.error(f"Eroare la calcularea scorului de calitate: {e}")
    st.stop()

# --- 2. TABURILE APLICATIEI ---
# ******************* MODIFICARE AICI (Adăugat Tab 4) *******************
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Legătura Atribute - Preț", 
    "📈 Evoluția Prețurilor", 
    "⭐ Raport Preț-Calitate", 
    "⚖️ Comparator Direct", 
    "🔮 Estimare Preț"
])
# *******************************************************************


# --- TAB 1: Analiza de Corelație ---
with tab1:
    st.header("1. Determinarea Legăturii dintre Atribute și Preț")
    
    colA, colB = st.columns([1, 1])

    # 1.1 Tabel de Corelații
    numeric_df = df[numeric_features]
    correlation_matrix = numeric_df.corr().round(2)
    
    with colA:
        st.subheader("Tabel de Corelații cu Prețul")
        st.dataframe(correlation_matrix[['Price']].sort_values(by='Price', ascending=False))
        st.markdown("**Interpretare:** Valori apropiate de 1 (corelație directă) sau -1 (corelație inversă) indică o legătură puternică.")
        
    # 1.2 Grafic de Relație Preț vs. Atribut
    with colB:
        st.subheader("Vizualizarea Relației")
        feature = st.selectbox("Alegeți atributul de analizat:", 
                               ['Engine_Volume', 'Mileage_Num', 'tax', 'mpg', 'Fuel_Type', 'Gear_Box_Type'])

        if feature in ['Engine_Volume', 'Mileage_Num', 'tax', 'mpg']:
            fig = px.scatter(df, x=feature, y='Price', color='Manufacturer', 
                             title=f'Preț vs. {feature} (cu Linie de Trend OLS)', 
                             hover_data=['model', 'Year'],
                             trendline='ols',
                             trendline_scope="overall")
        else: 
            fig = px.violin(df, x=feature, y='Price', color=feature, 
                             box=True, 
                             points="outliers", 
                             title=f'Distribuția DENSITĂȚII Prețului pe Clase de {feature}')

        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: Evoluția Prețurilor ---
with tab2:
    st.header("2. Reprezentarea Grafică a Evoluției Prețurilor")

    colC, colD, colE = st.columns(3) 
    
    selected_manufacturer = colC.selectbox("Selectați Marca:", sorted(df['Manufacturer'].unique()))
    
    model_options = sorted(df[df['Manufacturer'] == selected_manufacturer]['model'].unique())
    selected_model = colD.selectbox("Selectați Modelul:", model_options)
    
    df_temp = df[(df['Manufacturer'] == selected_manufacturer) & (df['model'] == selected_model)].copy()
    engine_options_tab2 = sorted(df_temp['Engine_Volume'].dropna().unique())
    
    engine_options_all = ['Toate'] + [str(e) for e in engine_options_tab2]
    selected_engine = colE.selectbox("Selectați Motorizarea (Volum):", engine_options_all)
    
    df_filtered = df_temp.copy()
    
    if selected_engine != 'Toate':
        engine_val = float(selected_engine) 
        df_filtered = df_filtered[df_filtered['Engine_Volume'] == engine_val]

    price_evolution = df_filtered.groupby('Year')['Price'].median().reset_index()

    if not price_evolution.empty and price_evolution.shape[0] > 1:
        title = f'Evoluția Prețului Median pentru {selected_manufacturer} {selected_model} (Motor: {selected_engine})'
        
        fig_evol = px.line(price_evolution, x='Year', y='Price', markers=True, 
                             title=title,
                             labels={'Year': 'Anul de Producție', 'Price': 'Preț Median ($)'})
        st.plotly_chart(fig_evol, use_container_width=True)
        st.markdown(f"> **Observație:** Acest grafic arată deprecierea modelului `{selected_model}` de-a lungul timpului. (Motor: `{selected_engine}`).")
    else:
        st.warning(f"Nu există suficiente date (minim 2 ani) pentru {selected_manufacturer} {selected_model} cu Motorizarea `{selected_engine}` pentru a arăta o evoluție concludentă.")


# --- TAB 3: Ierarhizarea Preț-Calitate ---
with tab3:
    st.header("3. Ierarhizarea Modelelor și Recomandarea de Cumpărare")
    
    tab3_col1, tab3_col2 = st.columns([1, 1])

    # --- 3.1 CLASAMENT GENERAL ---
    with tab3_col1:
        st.subheader("Clasament General (Raport Calitate/Preț)")
        st.markdown(f"*(Scorul de Calitate este definit pe baza Anului, Rulajului, MPG și Volumului Motorului.)*")
        
        display_cols = ['Manufacturer', 'model', 'Raport_Pret_Calitate_Mediu', 'Scor_Calitate_Mediu', 'Pret_Mediu', 'Numar_Inregistrari']
        st.dataframe(ranking_df[display_cols].head(1000).style.format(
            {'Raport_Pret_Calitate_Mediu': '{:.2f}', 'Scor_Calitate_Mediu': '{:.1f}', 'Pret_Mediu': '${:,.0f}'}
        ), hide_index=True, use_container_width=True, height=600)
        st.markdown("> **Un raport mai mare** indică un model care oferă mai multă 'calitate' per unitate monetară.")

    # --- 3.2 RECOMANDARE PERSONALIZATĂ ---
    with tab3_col2:
        st.subheader("💸 Căutare după Buget (Recomandare)")
        
        max_price = int(df['Price'].max())
        
        buget_min_max = st.slider(
            "Selectați Intervalul de Buget (USD):", 
            min_value=0, 
            max_value=max_price, 
            value=(5000, min(20000, max_price)), 
            step=500
        )
        buget_min = buget_min_max[0]
        buget_max = buget_min_max[1]
        
        df_buget = df[(df['Price'] >= buget_min) & (df['Price'] <= buget_max)].copy()
        
        if df_buget.empty:
            st.warning(f"Niciun autoturism din baza de date nu se încadrează în intervalul ${buget_min:,.0f} - ${buget_max:,.0f}.")
        else:
            ranking_buget = ranking_df[
                (ranking_df['Pret_Mediu'] >= buget_min) & 
                (ranking_df['Pret_Mediu'] <= buget_max)
            ].sort_values(
                by='Raport_Pret_Calitate_Mediu', ascending=False
            )
            
            st.info(f"**Top 3 Modele** cu cel mai bun raport C/P (Preț mediu între ${buget_min:,.0f} și ${buget_max:,.0f}):")
            if not ranking_buget.empty:
                rec_model_cols = ['Manufacturer', 'model', 'Raport_Pret_Calitate_Mediu', 'Pret_Mediu']
                st.dataframe(ranking_buget[rec_model_cols].head(3).style.format(
                    {'Raport_Pret_Calitate_Mediu': '{:.2f}', 'Pret_Mediu': '${:,.0f}'}
                ), hide_index=True)
            else:
                 st.markdown("Nu s-au găsit modele medii care să se încadreze în acest interval de buget.")

            st.markdown("---")

            best_car = df_buget.sort_values(by='Quality_Score', ascending=False).iloc[0]

            st.success(f"**Cea Mai Bună Ofertă Găsită** (Scor de Calitate Maxim):")
            st.markdown(f"**Marca/Model:** `{best_car['Manufacturer']} {best_car['model']}`")
            st.markdown(f"**An:** `{best_car['Year']}`")
            st.markdown(f"**Preț:** **${best_car['Price']:,.0f}**")
            st.markdown(f"**Scor Calitate:** `{best_car['Quality_Score']:.1f}` (Raport C/P: `{best_car['Price_Quality_Ratio']:.2f}`)")


# *******************************************************************
# --- TAB 4 (NOU): Comparator Direct ---
# *******************************************************************
with tab4:
    st.header("⚖️ Comparator Direct Side-by-Side")
    st.markdown("Alegeți două modele (ex: finaliștii din Tab-ul 3) pentru a le compara direct.")
    
    comp_col1, comp_col2 = st.columns(2)
    
    # --- Selectoare Mașina 1 ---
    with comp_col1:
        st.subheader("Mașina 1")
        man1 = st.selectbox("Selectați Marca 1:", sorted(df['Manufacturer'].unique()), key='man1', index=0)
        
        model_options_1 = sorted(df[df['Manufacturer'] == man1]['model'].unique())
        mod1 = st.selectbox("Selectați Modelul 1:", model_options_1, key='mod1', index=0)

    # --- Selectoare Mașina 2 ---
    with comp_col2:
        st.subheader("Mașina 2")
        man2 = st.selectbox("Selectați Marca 2:", sorted(df['Manufacturer'].unique()), key='man2', index=1) # index 1 pt diferentiere
        
        model_options_2 = sorted(df[df['Manufacturer'] == man2]['model'].unique())
        mod2 = st.selectbox("Selectați Modelul 2:", model_options_2, key='mod2', index=0)

    st.markdown("---")

    # --- Afișare Comparație ---
    if st.button("Compară Modelele"):
        
        data1 = ranking_df[(ranking_df['Manufacturer'] == man1) & (ranking_df['model'] == mod1)]
        data2 = ranking_df[(ranking_df['Manufacturer'] == man2) & (ranking_df['model'] == mod2)]
        
        if data1.empty or data2.empty:
            st.error("Unul dintre modelele selectate nu are date suficiente. Încercați altele.")
        else:
            stats1 = data1.iloc[0]
            stats2 = data2.iloc[0]
            
            st.subheader("📈 Comparație Metrică (Medie)")
            
            viz_col1, viz_col2 = st.columns(2)
            
            # --- Carduri Mașina 1 ---
            with viz_col1:
                st.info(f"**{stats1['Manufacturer']} {stats1['model']}**")
                
                # Highlight diferențe (delta)
                delta_price = float(stats1['Pret_Mediu'] - stats2['Pret_Mediu'])
                delta_qual = float(stats1['Scor_Calitate_Mediu'] - stats2['Scor_Calitate_Mediu'])
                delta_ratio = float(stats1['Raport_Pret_Calitate_Mediu'] - stats2['Raport_Pret_Calitate_Mediu'])

                st.metric("Preț Mediu", f"${stats1['Pret_Mediu']:,.0f}", f"{delta_price:,.0f} $", help="Diferența față de Mașina 2")
                st.metric("Scor Calitate Mediu", f"{stats1['Scor_Calitate_Mediu']:.1f}", f"{delta_qual:.1f}", help="Diferența față de Mașina 2")
                st.metric("Raport Calitate/Preț", f"{stats1['Raport_Pret_Calitate_Mediu']:.2f}", f"{delta_ratio:.2f}", help="Diferența față de Mașina 2")

            # --- Carduri Mașina 2 ---
            with viz_col2:
                st.info(f"**{stats2['Manufacturer']} {stats2['model']}**")

                st.metric("Preț Mediu", f"${stats2['Pret_Mediu']:,.0f}", f"{-delta_price:,.0f} $", help="Diferența față de Mașina 1")
                st.metric("Scor Calitate Mediu", f"{stats2['Scor_Calitate_Mediu']:.1f}", f"{-delta_qual:.1f}", help="Diferența față de Mașina 1")
                st.metric("Raport Calitate/Preț", f"{stats2['Raport_Pret_Calitate_Mediu']:.2f}", f"{-delta_ratio:.2f}", help="Diferența față de Mașina 1")
            
            st.markdown("---")
            st.subheader("📊 Comparație Vizuală (Grafic Radar)")
            
            # --- Grafic Radar ---
            categories = ['Scor An (Nou)', 'Scor Consum (Economic)', 'Scor Rulaj (Mic)', 'Scor Motor (Eficient)']
            
            values1 = [
                stats1['Scor_An_Mediu'] * 100, 
                stats1['Scor_Consum_Mediu'] * 100, 
                stats1['Scor_Rulaj_Mediu'] * 100,
                stats1['Scor_Motor_Mediu'] * 100
            ]
            values2 = [
                stats2['Scor_An_Mediu'] * 100, 
                stats2['Scor_Consum_Mediu'] * 100, 
                stats2['Scor_Rulaj_Mediu'] * 100,
                stats2['Scor_Motor_Mediu'] * 100
            ]
            
            fig_radar = go.Figure()

            fig_radar.add_trace(go.Scatterpolar(
                  r=values1,
                  theta=categories,
                  fill='toself',
                  name=f"{man1} {mod1}"
            ))
            fig_radar.add_trace(go.Scatterpolar(
                  r=values2,
                  theta=categories,
                  fill='toself',
                  name=f"{man2} {mod2}"
            ))

            fig_radar.update_layout(
              polar=dict(
                radialaxis=dict(
                  visible=True,
                  range=[0, 100]
                )),
              showlegend=True,
              title="Comparație Atribute de Calitate (Scor 0-100)"
            )

            st.plotly_chart(fig_radar, use_container_width=True)


# *******************************************************************
# --- TAB 5 (fostul Tab 4): Estimare Preț ---
# *******************************************************************
with tab5:
    st.header("4. Estimarea Prețului")
    st.markdown("Modelul de Regresie Liniară este folosit pentru predicția prețului pe baza caracteristicilor introduse.")

    # --- MODEL TRAINING (BLOC COMUN) ---
    @st.cache_resource
    def train_model():
        features = ['Year', 'Mileage_Num', 'Engine_Volume', 'Gear_Box_Type', 'Manufacturer', 'model', 'tax', 'mpg', 'Fuel_Type']
        
        # Filtrează df-ul DOAR pentru antrenare
        train_df = df.dropna(subset=features)
        X = train_df[features]
        y = train_df['Price']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gear_Box_Type', 'Manufacturer', 'model', 'Fuel_Type'])
            ],
            remainder='passthrough'
        )
        
        if X.empty:
            return None, None
            
        X_processed = preprocessor.fit_transform(X)
        
        if X_processed.shape[0] < 2:
             return None, None
        
        try:
            model = LinearRegression()
            model.fit(X_processed, y)
            return model, preprocessor
        except Exception as e:
            st.error(f"Eroare la antrenarea modelului: {e}.")
            return None, None

    model, preprocessor = train_model()
    
    if model is None or preprocessor is None:
        st.error("Nu există suficiente date pentru a antrena modelul de predicție.")
        st.stop()
    # -----------------------

    st.subheader("I. Estimarea Prețului pentru Autoturism Existent")
    
    # --- COLECTAREA INPUTULUI UTILIZATORULUI (EXISTENT) ---
    colE, colF, colG = st.columns(3)
    
    max_year_data = int(df['Year'].max()) 
    input_year = colE.slider(
        "Anul de Producție:", 
        int(df['Year'].min()), 
        max_year_data + 3, 
        max_year_data 
    )
    input_mileage = colF.number_input("Kilometraj (km):", min_value=0, max_value=int(df['Mileage_Num'].max()) if not df['Mileage_Num'].empty else 300000, value=10000)
    
    engine_options_exist = sorted(df['Engine_Volume'].dropna().unique())
    input_engine = colG.selectbox("Volumul Motorului:", engine_options_exist, index=min(1, len(engine_options_exist)-1) if len(engine_options_exist)>0 else 0)
    
    tax_options_exist = sorted(df['tax'].dropna().unique())
    input_tax = colE.selectbox("Taxă (tax):", tax_options_exist, index=0)
    
    mpg_options_exist = sorted(df['mpg'].dropna().unique())
    input_mpg = colF.selectbox("Consum (mpg):", mpg_options_exist, index=0)

    input_manufacturer = colG.selectbox("Marca:", sorted(df['Manufacturer'].unique()), index=0)
    
    model_options_exist = sorted(df[df['Manufacturer'] == input_manufacturer]['model'].unique())
    input_model = colG.selectbox("Modelul:", model_options_exist, index=0, key='input_model_exist')
    
    fuel_options_exist = sorted(df['Fuel_Type'].unique())
    input_fuel = colE.selectbox("Tip Combustibil (Fuel Type):", fuel_options_exist, index=0)
    
    gear_options_exist = sorted(df['Gear_Box_Type'].unique())
    input_gear = colF.selectbox("Tip Cutie Viteze:", gear_options_exist, index=0)
    
    if st.button("Estimează Prețul (Exemplar Existent)"):
        
        new_data = pd.DataFrame([{
            'Year': input_year, 'Mileage_Num': input_mileage, 'Engine_Volume': input_engine, 
            'Gear_Box_Type': input_gear, 'Manufacturer': input_manufacturer, 'model': input_model,
            'tax': input_tax, 'mpg': input_mpg, 'Fuel_Type': input_fuel
        }])
        
        try:
            new_data_processed = preprocessor.transform(new_data)
        except ValueError as e:
            st.error(f"Eroare de preprocesare: {e}. Asigurați-vă că ați selectat valori valide.")
            st.stop()
        
        predicted_price = model.predict(new_data_processed)[0]
        
        st.success(f"**Prețul estimat pentru acest autoturism ({input_manufacturer} {input_model}) este:**")
        st.info(f"## ${predicted_price:,.0f}")


    st.markdown("---")
    st.subheader("II. Extrapolarea Prețului pentru Modele 2026 (Viitor)")
    st.warning("Atenție! Aceasta este o extrapolare statistică, precizia poate fi redusă.")

    # --- COLECTAREA INPUTULUI UTILIZATORULUI (2026) ---
    colE_26, colF_26, colG_26 = st.columns(3)

    input_year_26 = 2026
    colE_26.markdown(f"**Anul de Producție: {input_year_26}**") 

    input_manufacturer_26 = colF_26.selectbox(
        "Marca (2026):", 
        sorted(df['Manufacturer'].unique()), 
        key='man_26', 
        index=0
    )

    model_options_26 = sorted(df[df['Manufacturer'] == input_manufacturer_26]['model'].unique())
    input_model_26 = colG_26.selectbox(
        "Modelul (2026):", 
        model_options_26, 
        key='model_26', 
        index=0
    )

    colH_26, colI_26, colJ_26 = st.columns(3)

    input_mileage_26 = 100 
    colH_26.markdown(f"Kilometraj (Estimat pentru un model nou): **{input_mileage_26} km**") 

    df_temp_26 = df[(df['Manufacturer'] == input_manufacturer_26) & (df['model'] == input_model_26)].copy()
    engine_options_26 = sorted(df_temp_26['Engine_Volume'].dropna().unique())
    if not engine_options_26:
          engine_options_26 = sorted(df['Engine_Volume'].dropna().unique()) 
          st.warning(f"Nu există date de motorizare pentru combinația {input_manufacturer_26} {input_model_26}. Se folosesc toate volumele motorului.")

    input_engine_26 = colI_26.selectbox(
        "Volum Motor (2026):", 
        engine_options_26, 
        key='engine_26', 
        index=min(1, len(engine_options_26)-1) if len(engine_options_26)>0 else 0
    )

    input_fuel_26 = colJ_26.selectbox("Tip Combustibil (2026):", fuel_options_exist, key='fuel_26', index=0)

    colK_26, colL_26, colM_26 = st.columns(3)

    input_gear_26 = colK_26.selectbox("Cutie Viteze (2026):", gear_options_exist, key='gear_26', index=0)
    
    median_tax = df['tax'].median()
    median_mpg = df['mpg'].median()
    
    colL_26.markdown(f"Taxă (Est.): **{median_tax:.0f}**")
    colM_26.markdown(f"Consum MPG (Est.): **{median_mpg:.1f}**")

    if st.button(f"Estimează Prețul pentru Modelul NOU 2026"):

        new_data_26 = pd.DataFrame([{
            'Year': input_year_26,
            'Mileage_Num': input_mileage_26, 
            'Engine_Volume': input_engine_26,
            'Gear_Box_Type': input_gear_26,
            'Manufacturer': input_manufacturer_26,
            'model': input_model_26, 
            'tax': median_tax, 
            'mpg': median_mpg, 
            'Fuel_Type': input_fuel_26
        }])

        
        try:
            new_data_processed_26 = preprocessor.transform(new_data_26)
        except ValueError as e:
            st.error(f"Eroare de preprocesare: {e}. Asigurați-vă că ați selectat valori valide.")
            st.stop()
        
        predicted_price_26 = model.predict(new_data_processed_26)[0]
        
        st.success(f"**Prețul estimat pentru {input_manufacturer_26} {input_model_26} din {input_year_26} este:**")
        st.info(f"## ${predicted_price_26:,.0f}")
        
        st.markdown(f"*Această estimare se bazează pe tendințele pieței pentru **{input_manufacturer_26}** și caracteristicile introduse.*")
        
        # 1. Calculează datele istorice (similar Tab 2)
        df_model_history = df_temp_26.groupby('Year')['Price'].median().reset_index()

        if not df_model_history.empty:
            
            # 2. Adaugă punctul prezis
            new_point = pd.DataFrame([{
                'Year': input_year_26, 
                'Price': predicted_price_26,
                'Type': 'Predicție 2026'
            }])
            
            # 3. Pregătește datele istorice pentru grafic
            df_model_history['Type'] = 'Istoric Median'
            
            # 4. Combină datele
            df_combined = pd.concat([df_model_history, new_point], ignore_index=True)

            # 5. Creează figura (Linia pentru Istoric)
            fig_pred = go.Figure()
            
            # Adaugă linia istorică
            fig_pred.add_trace(go.Scatter(
                x=df_combined[df_combined['Type'] == 'Istoric Median']['Year'],
                y=df_combined[df_combined['Type'] == 'Istoric Median']['Price'],
                mode='lines+markers',
                name='Preț Median Istoric',
                line=dict(color='blue'),
                marker=dict(size=8, color='blue')
            ))

            # Adaugă punctul prezis (marker mare, roșu)
            fig_pred.add_trace(go.Scatter(
                x=new_point['Year'],
                y=new_point['Price'],
                mode='markers',
                name=f'Predicție {input_year_26}',
                marker=dict(size=15, color='red', symbol='star'),
                hovertext=f"Preț estimat: ${predicted_price_26:,.0f}"
            ))
            
            # Setează titlul și etichetele
            fig_pred.update_layout(
                title=f'Evoluția Prețului Median cu Extrapolarea {input_year_26} pentru {input_manufacturer_26} {input_model_26}',
                xaxis_title='Anul de Producție',
                yaxis_title='Preț Median ($)',
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)

        else:
            st.warning("Nu există date istorice suficiente pentru acest model pentru a afișa graficul de evoluție.")