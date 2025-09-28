import streamlit as st
st.set_page_config(page_title=" 🇮🇳 Samundra Setu", layout="wide")

st.markdown("""
<style>
#bgvideo {
  position: fixed;
  right: 0;
  bottom: 0;
  min-width: 100vw;
  min-height: 100vh;
  object-fit: cover;
  z-index: -1;
}
.overlay {
  position: fixed;
  top: 20%;
  width: 100vw;
  text-align: center;
  color: white;
  z-index: 1;
  animation: fadeout 1s ease 10s forwards; /* fade out after 10s */
}
@keyframes fadeout {
  to { opacity: 0; visibility: hidden; }
}
.overlay h1 {
  font-size: 60px;
  color: orange;
  font-weight: bold;
  text-shadow: 2px 2px 6px black;
}
.overlay p {
  font-size: 24px;
  color: #f0f0f0;
  text-shadow: 1px 1px 4px black;
}
</style>

<video autoplay muted loop id="bgvideo">
  <source src="s1.mp4" type="video/mp4">
</video>

<div class="overlay" id="samundra-overlay">
  <h1>Samundra Setu</h1>
  <p>You are Samundra AI, a helpful oceanographic assistant with expertise in marine data analysis.</p>
</div>
""", unsafe_allow_html=True)

st.video("s1.mp4",loop = True)

tab1, tab2, tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(["Samundra AI", "Ocean-Current", "Live INCOIS data","New LLM MODEL","Alerts","Satellite Imagery and isro information","Contact Us","About Us"])
with tab1:
    
    import streamlit as st
    import pandas as pd
    import requests
    import pyttsx3
    import speech_recognition as sr
    from threading import Thread, Lock
    import sqlite3
    import time
    import folium
    from streamlit_folium import st_folium
    from folium.plugins import MarkerCluster
    import re
    import queue
    import threading
    import matplotlib.pyplot as plt
    import numpy as np
    from math import radians, sin, cos, sqrt, atan2


    if 'question_queue' not in st.session_state:
        st.session_state.question_queue = queue.Queue()
    if 'listening' not in st.session_state:
        st.session_state.listening = True
    if 'tts_engine' not in st.session_state:
        try:
            st.session_state.tts_engine = pyttsx3.init()
        except RuntimeError:
            import pyttsx3.drivers
            st.session_state.tts_engine = pyttsx3.drivers.sapi5.init()


    MISTRAL_API_KEY = 'nZwqy7s0iceaClHw0TfpW5sCw2mPeVta'
    MISTRAL_MODEL = 'mistral-large-2411'
    MISTRAL_URL = 'https://api.mistral.ai/v1/chat/completions'

    def ask_mistral(question, context=""):
        headers = {
            'Authorization': f'Bearer {MISTRAL_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': MISTRAL_MODEL,
            'messages': [
                {'role': 'system', 'content': 'You are Samundra AI, a helpful oceanographic assistant with expertise in marine data analysis. Provide detailed, scientific answers about oceanographic data including temperature, salinity, pressure profiles, and regional variations.'},
                {'role': 'user', 'content': f'{context}\nUser question: {question}'}
            ]
        }
        try:
            response = requests.post(MISTRAL_URL, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error contacting Mistral API: {e}"


    tts_lock = Lock()

    def speak(text):
        with tts_lock:
            try:
                engine = st.session_state.tts_engine
                engine.say(text)
                engine.runAndWait()
            except RuntimeError as e:
                st.error(f"Text-to-speech error: {e}")
            except Exception as e:
                st.error(f"Unexpected TTS error: {e}")


    DB_PATH = "incois_2025 (8).db"

    def load_database(path=DB_PATH):
        try:
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if not tables:
                st.error("No tables found in the database!")
                return None, None
            table_name = tables[0][0]
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()
            return df, table_name
        except Exception as e:
            st.error(f"Error loading database: {e}")
            return None, None

    df, table_name = load_database()
    if df is not None:
        st.write(f"Loaded {len(df)} rows from table `{table_name}`")

    def create_profile_summary(df):
        if 'profile_date' in df.columns:
            df['profile_date_str'] = df['profile_date'].astype(str)
            
            df['profile_date_clean'] = df['profile_date_str'].apply(
                lambda x: x.split('.')[0] if '.' in x else x
            )
            
            df['profile_date'] = pd.to_datetime(
                df['profile_date_clean'], 
                format='%Y-%m-%d %H:%M:%S', 
                errors='coerce'
            )
            
            df = df.drop(['profile_date_str', 'profile_date_clean'], axis=1, errors='ignore')
        
        profile_summary = df.groupby(['float_id', 'profile_date']).agg({
            'latitude': 'first',
            'longitude': 'first',
            'file': 'first',
            'pressure_dbar': ['min', 'max', 'count'],
            'temperature': ['min', 'max', 'mean'],
            'salinity': ['min', 'max', 'mean']
        }).reset_index()
        
        profile_summary.columns = ['_'.join(col).strip('_') for col in profile_summary.columns.values]
        
        profile_summary = profile_summary.rename(columns={
            'float_id_': 'float_id',
            'profile_date_': 'profile_date',
            'latitude_first': 'latitude',
            'longitude_first': 'longitude',
            'file_first': 'file',
            'pressure_dbar_min': 'min_pressure',
            'pressure_dbar_max': 'max_pressure',
            'pressure_dbar_count': 'measurement_count',
            'temperature_min': 'min_temp',
            'temperature_max': 'max_temp',
            'temperature_mean': 'mean_temp',
            'salinity_min': 'min_salinity',
            'salinity_max': 'max_salinity',
            'salinity_mean': 'mean_salinity'
        })
        
        return profile_summary

    if df is not None:
        profile_summary = create_profile_summary(df)
    else:
        profile_summary = pd.DataFrame()


    def haversine_distance(lat1, lon1, lat2, lon2):
    
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        r = 6371  
        return c * r

    def find_nearest_profiles(lat, lon, profiles_df, n=5):
        
        distances = []
        for idx, row in profiles_df.iterrows():
            dist = haversine_distance(lat, lon, row['latitude'], row['longitude'])
            distances.append(dist)
        
        result_df = profiles_df.copy()
        result_df['distance_km'] = distances
        
        return result_df.nsmallest(n, 'distance_km')

    KNOWN_LOCATIONS = {
        'chennai': {'lat': 13.0827, 'lon': 80.2707},
        'mumbai': {'lat': 19.0760, 'lon': 72.8777},
        'kolkata': {'lat': 22.5726, 'lon': 88.3639},
        'kochi': {'lat': 9.9312, 'lon': 76.2673},
        'visakhapatnam': {'lat': 17.6868, 'lon': 83.2185},
        'goa': {'lat': 15.2993, 'lon': 74.1240},
        'andaman': {'lat': 11.7401, 'lon': 92.6586},
        'lakshadweep': {'lat': 10.5667, 'lon': 72.6417}
    }


    def get_region_stats(profiles_df, region_name):
        """Get statistics for a specific region"""
        region_data = filter_profiles_by_region(profiles_df, region_name)
        
        if region_data.empty:
            return f"No data available for {region_name}"
        
        stats = {
            'profile_count': len(region_data),
            'temp_range': f"{region_data['min_temp'].min():.2f} to {region_data['max_temp'].max():.2f} °C",
            'salinity_range': f"{region_data['min_salinity'].min():.2f} to {region_data['max_salinity'].max():.2f} PSU",
            'pressure_range': f"{region_data['min_pressure'].min():.1f} to {region_data['max_pressure'].max():.1f} dbar",
            'avg_measurements': f"{region_data['measurement_count'].mean():.1f}",
            'time_range': f"{region_data['profile_date'].min().strftime('%Y-%m-%d')} to {region_data['profile_date'].max().strftime('%Y-%m-%d')}" if not region_data.empty else "N/A"
        }
        
        return stats

    def get_salinity_analysis(df):
        if df.empty:
            return "No data available for salinity analysis"
        
        max_salinity = df['salinity'].max()
        max_salinity_locations = df[df['salinity'] == max_salinity][['latitude', 'longitude', 'pressure_dbar']].head(5)
        
        regions = ['Bay of Bengal', 'Arabian Sea', 'Indian Ocean']
        region_salinity = {}
        
        for region in regions:
            region_data = filter_profiles_by_region(profile_summary, region)
            if not region_data.empty:
                region_salinity[region] = region_data['mean_salinity'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if region_salinity:
            ax.bar(region_salinity.keys(), region_salinity.values())
            ax.set_ylabel('Average Salinity (PSU)')
            ax.set_title('Average Salinity by Region')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        analysis = f"""
        Salinity Analysis:
        - Maximum salinity recorded: {max_salinity:.2f} PSU
        - Locations with highest salinity: 
        """
        
        for idx, row in max_salinity_locations.iterrows():
            analysis += f"\n  - Latitude: {row['latitude']:.4f}, Longitude: {row['longitude']:.4f} at {row['pressure_dbar']} dbar"
        
        for region, avg_salinity in region_salinity.items():
            analysis += f"\n- Average salinity in {region}: {avg_salinity:.2f} PSU"
        
        return analysis

    def get_profile_details(float_id, profile_date=None):
        if profile_date:
            profile_data = df[(df['float_id'] == float_id) & (df['profile_date'] == profile_date)]
        else:
            # Get the most recent profile for this float
            recent_date = df[df['float_id'] == float_id]['profile_date'].max()
            profile_data = df[(df['float_id'] == float_id) & (df['profile_date'] == recent_date)]
        
        if profile_data.empty:
            return "No data found for the specified profile"
        
        details = {
            'float_id': float_id,
            'profile_date': profile_data['profile_date'].iloc[0],
            'location': f"{profile_data['latitude'].iloc[0]:.4f}, {profile_data['longitude'].iloc[0]:.4f}",
            'measurement_count': len(profile_data),
            'max_depth': f"{profile_data['pressure_dbar'].max():.1f} dbar",
            'surface_temp': f"{profile_data[profile_data['pressure_dbar'] == profile_data['pressure_dbar'].min()]['temperature'].iloc[0]:.2f} °C",
            'surface_salinity': f"{profile_data[profile_data['pressure_dbar'] == profile_data['pressure_dbar'].min()]['salinity'].iloc[0]:.2f} PSU"
        }
        
        return details


    def create_map(profiles_df, center=[20, 80], zoom_start=3):
        m = folium.Map(location=center, zoom_start=zoom_start, tiles='OpenStreetMap')
        marker_cluster = MarkerCluster().add_to(m)
        
        for idx, row in profiles_df.iterrows():
            popup_content = f"""
            <div style="width: 250px;">
                <h4 style="margin-bottom: 5px;">Float {row['float_id']}</h4>
                <p style="margin: 2px 0;"><b>Date:</b> {row['profile_date']}</p>
                <p style="margin: 2px 0;"><b>Location:</b> {row['latitude']:.4f}, {row['longitude']:.4f}</p>
                <p style="margin: 2px 0;"><b>Measurements:</b> {row['measurement_count']}</p>
                <p style="margin: 2px 0;"><b>Pressure Range:</b> {row['min_pressure']:.1f} - {row['max_pressure']:.1f} dbar</p>
                <p style="margin: 2px 0;"><b>Temperature:</b> {row['min_temp']:.2f} - {row['max_temp']:.2f} °C</p>
                <p style="margin: 2px 0;"><b>Salinity:</b> {row['min_salinity']:.2f} - {row['max_salinity']:.2f} PSU</p>
            </div>
            """
            
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"Float {row['float_id']} - {row['profile_date']}",
                icon=folium.Icon(color='blue', icon='tint', prefix='fa')
            ).add_to(marker_cluster)
        
        return m


    def filter_profiles_by_region(profiles_df, region_name):
        region_bounds = {
            'bay of bengal': {'lat_min': 5, 'lat_max': 22, 'lon_min': 80, 'lon_max': 95},
            'arabian sea': {'lat_min': 5, 'lat_max': 25, 'lon_min': 50, 'lon_max': 75},
            'indian ocean': {'lat_min': -30, 'lat_max': 30, 'lon_min': 20, 'lon_max': 120},
            'south china sea': {'lat_min': 0, 'lat_max': 25, 'lon_min': 105, 'lon_max': 121},
            'andaman sea': {'lat_min': 5, 'lat_max': 15, 'lon_min': 92, 'lon_max': 98},
            'north indian ocean': {'lat_min': 0, 'lat_max': 30, 'lon_min': 40, 'lon_max': 100}
        }
        
        region_name = region_name.lower()
        if region_name in region_bounds:
            bounds = region_bounds[region_name]
            filtered = profiles_df[
                (profiles_df['latitude'] >= bounds['lat_min']) &
                (profiles_df['latitude'] <= bounds['lat_max']) &
                (profiles_df['longitude'] >= bounds['lon_min']) &
                (profiles_df['longitude'] <= bounds['lon_max'])
            ]
            return filtered
        else:
            return profiles_df


    def listen_loop():
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while st.session_state.listening:
            try:
                with mic as source:
                    audio = recognizer.listen(source, phrase_time_limit=5)
                text = recognizer.recognize_google(audio).lower()
                if "hey samundra ai" in text or "hey samundra" in text or "samundra ai" in text:
                    st.session_state.question_queue.put("Wake word detected!")
                    speak("Yes, I am listening")
                    # Listen for the actual question
                    with mic as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio2 = recognizer.listen(source, phrase_time_limit=10)
                    q = recognizer.recognize_google(audio2)
                    st.session_state.question_queue.put(q)
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                st.error(f"Speech recognition error: {e}")
            except Exception as e:
                st.error(f"Unexpected error in voice recognition: {e}")
            time.sleep(0.5)

    if st.session_state.listening:
        voice_thread = Thread(target=listen_loop, daemon=True)
        voice_thread.start()


    st.title("Samundra AI 🌊 - Ocean Data Assistant")
    st.write("Exploring oceanographic profiles with interactive mapping")

    if not profile_summary.empty:
        st.header("Oceanographic Profiles Map")
        
        regions = ['All Regions', 'Bay of Bengal', 'Arabian Sea', 'Indian Ocean', 'South China Sea', 'Andaman Sea', 'North Indian Ocean']
        selected_region = st.selectbox("Filter by region:", regions, index=0)
        
        if selected_region != 'All Regions':
            filtered_profiles = filter_profiles_by_region(profile_summary, selected_region)
            region_center = {
                'Bay of Bengal': [15, 87.5],
                'Arabian Sea': [15, 65],
                'Indian Ocean': [0, 70],
                'South China Sea': [12.5, 113],
                'Andaman Sea': [10, 95],
                'North Indian Ocean': [15, 75]
            }
            center = region_center.get(selected_region, [20, 80])
            zoom = 5
        else:
            filtered_profiles = profile_summary
            center = [20, 80]
            zoom = 3
        
        st.write(f"Showing {len(filtered_profiles)} profiles in {selected_region}")
        
        ocean_map = create_map(filtered_profiles, center=center, zoom_start=zoom)
        st_folium(ocean_map, width=700, height=500)
        
        st.subheader("Profile Summary")
        st.dataframe(filtered_profiles[['float_id', 'profile_date', 'latitude', 'longitude', 'measurement_count']].head(10))
        
        if selected_region != 'All Regions':
            stats = get_region_stats(profile_summary, selected_region)
            st.subheader(f"Statistics for {selected_region}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Profiles", stats['profile_count'])
                st.metric("Temperature Range", stats['temp_range'])
            with col2:
                st.metric("Salinity Range", stats['salinity_range'])
                st.metric("Avg Measurements", stats['avg_measurements'])
            with col3:
                st.metric("Pressure Range", stats['pressure_range'])
                st.metric("Time Range", stats['time_range'])
    else:
        st.warning("No profile data available for mapping.")

    st.header("Ask a Question")
    user_question = st.text_input("Type your question about the oceanographic data:")

    if user_question:
        st.session_state.question_queue.put(user_question)

    if not st.session_state.question_queue.empty():
        current_q = st.session_state.question_queue.get()
        st.info(f"Question: {current_q}")
        
        context = f"""
        Oceanographic profiles data loaded from database. 
        There are {len(profile_summary) if not profile_summary.empty else 0} unique profiles in the dataset.
        The data includes measurements of pressure, temperature, and salinity at various depths.
        """
        
        location_mentioned = None
        for location in KNOWN_LOCATIONS:
            if location in current_q.lower():
                location_mentioned = location
                break
        
        if location_mentioned and not profile_summary.empty:
            loc_data = KNOWN_LOCATIONS[location_mentioned]
            nearest_profiles = find_nearest_profiles(loc_data['lat'], loc_data['lon'], profile_summary, n=3)
            
            context += f"\nFor {location_mentioned.title()} (Lat: {loc_data['lat']}, Lon: {loc_data['lon']}), the nearest profiles are:"
            
            for idx, row in nearest_profiles.iterrows():
                context += f"\n- Float {row['float_id']} at {row['distance_km']:.1f} km away (Lat: {row['latitude']:.4f}, Lon: {row['longitude']:.4f})"
                context += f"\n  Profile date: {row['profile_date']}, Measurements: {row['measurement_count']}"
                context += f"\n  Temperature: {row['mean_temp']:.2f} °C, Salinity: {row['mean_salinity']:.2f} PSU"
        
        if 'salinity' in current_q.lower():
            salinity_analysis = get_salinity_analysis(df)
            context += f"\n{salinity_analysis}"
        
        region_mentioned = None
        for region in regions[1:]:  # Skip 'All Regions'
            if region.lower() in current_q.lower():
                region_mentioned = region
                break
        
        if region_mentioned and not profile_summary.empty:
            region_data = filter_profiles_by_region(profile_summary, region_mentioned)
            context += f"\nFor the {region_mentioned}, there are {len(region_data)} profiles available."
            if not region_data.empty:
                context += f"\nThe profiles in {region_mentioned} range from {region_data['min_temp'].min():.2f} to {region_data['max_temp'].max():.2f} °C in temperature."
                context += f"\nSalinity values range from {region_data['min_salinity'].min():.2f} to {region_data['max_salinity'].max():.2f} PSU."
                context += f"\nPressure measurements range from {region_data['min_pressure'].min():.1f} to {region_data['max_pressure'].max():.1f} dbar."
        
        float_id_match = re.search(r'float\s*(\d+)', current_q.lower())
        if float_id_match and not profile_summary.empty:
            float_id = int(float_id_match.group(1))
            float_data = profile_summary[profile_summary['float_id'] == float_id]
            if not float_data.empty:
                context += f"\nFor float {float_id}, there are {len(float_data)} profiles available."
                context += f"\nThe most recent profile was on {float_data['profile_date'].max()}."
        
        response_text = ask_mistral(current_q, context=context)
        st.success(response_text)
        
        tts_thread = Thread(target=speak, args=(response_text,), daemon=True)
        tts_thread.start()

    st.subheader("Try asking:")
    st.markdown("""
    - Where is salinity highest in the data?
    - Show me profiles in the Bay of Bengal
    - What's the temperature range in the Arabian Sea?
    - How many measurements were taken in the Indian Ocean?
    - Tell me about the salinity patterns in Float 1902669
    - Compare temperature between Bay of Bengal and Arabian Sea
    - What is the maximum depth recorded in the data?
    - Nearest profile near Chennai
    - Profiles closest to Mumbai
    - Ocean data near Kolkata
    """)


    st.markdown("---")
    st.markdown("Samundra AI - Oceanographic Data Analysis Assistant | Powered by Mistral AI")


    if st.button("Stop Voice Recognition"):
        st.session_state.listening = False
        st.write("Voice recognition stopped")

    if not profile_summary.empty:
        sizes = profile_summary.groupby('float_id')['measurement_count'].sum()
        labels = sizes.index.astype(str)
        explode = [0.05] * len(labels)  
        fig1, ax1 = plt.subplots(figsize=(1, 1))  
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal') 
        st.pyplot(fig1)
