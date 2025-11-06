import streamlit as st
import pandas as pd
import plotly.express as px
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
    
    df['Mileage_Num'] = pd.to_numeric(df['Mileage_Num'], errors='coerce')
    df['Engine_Volume'] = pd.to_numeric(df['Engine_Volume'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['tax'] = pd.to_numeric(df['tax'], errors='coerce')
    df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')

    required_cols = ['Price', 'Mileage_Num', 'Engine_Volume', 'mpg', 'tax']
    df.dropna(subset=required_cols, inplace=True)
    
    df['Year'] = df['Year'].astype(int)

except Exception as e:
    st.error(f"Eroare la încărcarea sau pregătirea datelor. Asigurați-vă că fișierul `CarsData.csv` există și are structura corectă: {e}")
    st.stop()
    
numeric_features = ['Price', 'Mileage_Num', 'Engine_Volume', 'Year', 'tax', 'mpg']

# --- CALCUL SCOR CALITATE ---
try:
    df['MPG_Score'] = (df['mpg'] - df['mpg'].min()) / (df['mpg'].max() - df['mpg'].min())
    
    min_eng = df['Engine_Volume'].min()
    max_eng = df['Engine_Volume'].max()
    df['Engine_Score'] = 1 - (df['Engine_Volume'] - min_eng) / (max_eng - min_eng)

    min_year = df['Year'].min()
    max_year = df['Year'].max()
    df['Year_Score'] = (df['Year'] - min_year) / (max_year - min_year)
    
    df['Quality_Score'] = (df['MPG_Score'] * 40) + (df['Engine_Score'] * 30) + (df['Year_Score'] * 30)
    
    df['Price_Quality_Ratio'] = df['Quality_Score'] / (df['Price'] / 1000)
    
    ranking_df = df.groupby(['Manufacturer', 'model']).agg(
        Pret_Mediu = ('Price', 'median'),
        Scor_Calitate_Mediu = ('Quality_Score', 'median'),
        Raport_Pret_Calitate_Mediu = ('Price_Quality_Ratio', 'median'),
        Numar_Inregistrari = ('Year', 'count')
    ).reset_index()
    
    ranking_df = ranking_df[ranking_df['Numar_Inregistrari'] >= 5]
    ranking_df = ranking_df.sort_values(by='Raport_Pret_Calitate_Mediu', ascending=False)
    
except Exception as e:
    st.error(f"Eroare la calcularea scorului de calitate: {e}")
    st.stop()


# --- 2. TABURILE APLICATIEI ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Legătura Atribute - Preț", "📈 Evoluția Prețurilor", "⭐ Raport Preț-Calitate", "🔮 Estimare Preț"])

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
        st.markdown(f"*(Scorul de Calitate este definit pe baza Anului, MPG și Volumului Motorului.)*")
        
        display_cols = ['Manufacturer', 'model', 'Raport_Pret_Calitate_Mediu', 'Scor_Calitate_Mediu', 'Pret_Mediu', 'Numar_Inregistrari']
        st.dataframe(ranking_df[display_cols].head(10).style.format(
            {'Raport_Pret_Calitate_Mediu': '{:.2f}', 'Scor_Calitate_Mediu': '{:.1f}', 'Pret_Mediu': '${:,.0f}'}
        ), hide_index=True, use_container_width=True)
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
            
# --- TAB 4: Estimare Preț ---
with tab4:
    st.header("4. Estimarea Prețului")
    st.markdown("Modelul de Regresie Liniară este folosit pentru predicția prețului pe baza caracteristicilor introduse.")

    # --- MODEL TRAINING (BLOC COMUN) ---
    # ******************* SCHIMBAREA AICI *******************
    features = ['Year', 'Mileage_Num', 'Engine_Volume', 'Gear_Box_Type', 'Manufacturer', 'model', 'tax', 'mpg', 'Fuel_Type']
    X = df.dropna(subset=features)[features]
    y = df.dropna(subset=features)['Price']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gear_Box_Type', 'Manufacturer', 'model', 'Fuel_Type'])
        ],
        remainder='passthrough'
    )
    # *******************************************************
    
    X_processed = preprocessor.fit_transform(X)
    
    if X_processed.shape[0] < 2:
        st.error("Nu există suficiente înregistrări pentru a antrena modelul de predicție.")
        st.stop()
    else:
        try:
            model = LinearRegression()
            model.fit(X_processed, y)
            
        except Exception as e:
            st.error(f"Eroare la antrenarea modelului: {e}. Vă rugați să verificați datele.")
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
    
    # ******************* SCHIMBAREA AICI: INPUT MODEL *******************
    model_options_exist = sorted(df[df['Manufacturer'] == input_manufacturer]['model'].unique())
    input_model = colG.selectbox("Modelul:", model_options_exist, index=0, key='input_model_exist')
    # *******************************************************************
    
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
            'model': input_model_26, # Adăugat model
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
        # ******************* SCHIMBAREA AICI: ELIMINAT TEXTUL DESPRE MODEL *******************
        st.markdown(f"*Această estimare se bazează pe tendințele pieței pentru **{input_manufacturer_26}** și caracteristicile introduse.*")
        # *************************************************************************************