import asyncio
import aiohttp
from collections import Counter
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from datetime import datetime, timedelta, timezone
import plotly.express as px
import plotly.graph_objects as go

# Farbzuordnung für Statuskategorien
status_colors = {
    "Termin gebucht": "#00c875",
    "Vorquali-Phase": "#579bfc",
    "Kein Interesse": "#c4c4c4",
    "Entscheidungsträger noch nicht erreicht": "#ffcc00"  # Optional: Fügen Sie eine Farbe für den zusätzlichen Status hinzu
}

# Funktion, um die URL zu formatieren
def format_url(original_url):
    base_url = "https://view.monday.com/board_data/"
    url_core = original_url.split('?')[0]
    formatted_url = f"{base_url}{url_core.split('/')[-1]}.json"
    return formatted_url

# Hilfsfunktion zur Umwandlung von Hex-Farben in RGBA
def hex_to_rgba(hex_color, alpha=0.3):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    if lv == 6:
        r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    elif lv == 3:
        r, g, b = hex_color[0]*2, hex_color[1]*2, hex_color[2]*2
    else:
        raise ValueError("Invalid hex color format")
    return f'rgba({int(r, 16)}, {int(g, 16)}, {int(b, 16)}, {alpha})'

# Funktion, um ein einzelnes Board zu analysieren
async def analyze_board(session, url, time_threshold):
    try:
        formatted_url = format_url(url)
        async with session.get(formatted_url) as response:
            if response.status == 200:
                board_data = await response.json()

                # Boardnamen extrahieren
                board_name = board_data.get('name', f"Unbekanntes Board ({formatted_url})")

                # Spalteninformationen extrahieren
                columns = board_data.get('board_data', {}).get('columns', [])

                # Dynamische Erkennung der Status-Spalte
                status_column = next(
                    (col for col in columns if col.get('type') == 'color' and 'labels' in col), None
                )

                if not status_column:
                    st.warning(f"Keine Status-Spalte im Board '{board_name}' gefunden.")
                    return {
                        "board_name": board_name,
                        "status_counts": {},
                        "status_changes": {},
                        "termin_gebuucht_dates": []
                    }

                # Labels aus der erkannten Status-Spalte extrahieren
                status_labels = status_column.get('labels', {})

                # Pulses extrahieren
                pulses = board_data.get('board_data', {}).get('pulses', [])

                # Dynamische Spalten-ID der Statuswerte
                status_column_id = status_column.get('id')

                # Statuscodes und Änderungsdaten extrahieren
                status_counts = Counter()
                status_changes = Counter()
                termin_gebuucht_dates = []

                # Relevante Status definieren
                relevant_statuses = {
                    "Termin gebucht": "Termin gebucht",
                    "Vorquali-Phase": "Vorquali-Phase",
                    "Kein Interesse": "Kein Interesse"
                }

                for pulse in pulses:
                    column_value = pulse['column_values'].get(status_column_id, {})
                    status_code = column_value.get('index')
                    changed_at = column_value.get('changed_at')

                    if status_code is not None:
                        raw_status = status_labels.get(str(status_code), "Unknown")
                        # Status umwandeln, falls nicht relevant
                        status = relevant_statuses.get(raw_status, "Entscheidungsträger noch nicht erreicht")
                        status_counts[status] += 1

                        # Änderungen verfolgen, falls ein Datum vorhanden ist
                        if changed_at:
                            try:
                                changed_at_dt = datetime.fromisoformat(changed_at.replace("Z", "+00:00"))
                                # Entferne die Zeitzone
                                changed_at_dt = changed_at_dt.replace(tzinfo=None)
                                if status == "Termin gebucht":
                                    termin_gebuucht_dates.append(changed_at_dt)
                                if changed_at_dt >= time_threshold:
                                    status_changes[status] += 1
                            except ValueError:
                                st.error(f"Ungültiges Datum im Board '{board_name}': {changed_at}")

                return {
                    "board_name": board_name,
                    "status_counts": dict(status_counts),
                    "status_changes": dict(status_changes),
                    "termin_gebuucht_dates": termin_gebuucht_dates
                }
            else:
                st.error(f"Fehler beim Abrufen des Boards: {url}, Status Code: {response.status}")
                return {
                    "board_name": f"Unbekanntes Board ({formatted_url})",
                    "status_counts": {},
                    "status_changes": {},
                    "termin_gebuucht_dates": []
                }
    except Exception as e:
        st.error(f"Fehler bei der Analyse des Boards: {url}, Fehler: {e}")
        return {
            "board_name": f"Unbekanntes Board ({url})",
            "status_counts": {},
            "status_changes": {},
            "termin_gebuucht_dates": []
        }

# Hauptfunktion zur Analyse mehrerer Boards
def get_start_of_current_week():
    today = datetime.now(timezone.utc)
    start_of_week = today - timedelta(days=today.weekday())  # Montag dieser Woche
    return datetime.combine(start_of_week, datetime.min.time())

async def analyze_multiple_boards(original_urls):
    time_threshold = get_start_of_current_week()  # Beginn der aktuellen Woche
    async with aiohttp.ClientSession() as session:
        tasks = [analyze_board(session, url, time_threshold) for url in original_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    dashboard_data = []
    all_termin_gebuucht_dates = []
    for result in results:
        if isinstance(result, Exception):
            continue
        board_name = result.get("board_name", "Unbekanntes Board")
        counts = result.get("status_counts", {}) or {}
        changes = result.get("status_changes", {}) or {}
        dates = result.get("termin_gebuucht_dates", []) or []
        dashboard_data.append({
            "board_name": board_name,
            "status_counts": counts,
            "status_changes": changes,
            "termin_gebuucht_dates": dates
        })
        all_termin_gebuucht_dates.extend(dates)
    return dashboard_data, all_termin_gebuucht_dates

# Funktion zur Zusammenführung ähnlicher Namen
def merge_similar_boards(data, similarity_threshold=85):
    merged_data = []
    processed = set()
    all_termin_gebuucht_dates = []

    for board in data:
        if board["board_name"] in processed:
            continue

        # Suche ähnliche Boardnamen
        matches = process.extract(
            board["board_name"],
            [b["board_name"] for b in data],
            scorer=fuzz.ratio,
        )

        # Filtere ähnliche Namen basierend auf dem Schwellenwert
        similar_boards = [
            match[0] for match in matches if match[1] >= similarity_threshold
        ]

        # Zusammenführen der Werte
        merged_board = {
            "board_name": board["board_name"],
            "status_counts": Counter(),
            "status_changes": Counter(),
            "termin_gebuucht_dates": []
        }
        for b_name in similar_boards:
            matching_board = next((b for b in data if b["board_name"] == b_name), None)
            if matching_board:
                merged_board["status_counts"].update(matching_board["status_counts"])
                merged_board["status_changes"].update(matching_board["status_changes"])
                merged_board["termin_gebuucht_dates"].extend(matching_board["termin_gebuucht_dates"])
                processed.add(b_name)

        merged_data.append(merged_board)
        all_termin_gebuucht_dates.extend(merged_board["termin_gebuucht_dates"])

    # Konvertiere Counters zurück zu normalen Dictionaries
    for board in merged_data:
        board["status_counts"] = dict(board["status_counts"])
        board["status_changes"] = dict(board["status_changes"])

    return merged_data, all_termin_gebuucht_dates

# Funktion zur Anzeige des Dashboards mit Plotly-Visualisierungen
def display_dashboard(data, all_termin_gebuucht_dates):
    try:
        st.title("Datenanalyse Dashboard")
        st.write("Tabellarische Übersicht der Boards:")

        # Erstelle eine Liste für die finalen Tabellenzeilen
        table_rows = []
        for board in data:
            row = {
                "board_name": board["board_name"],
                "Conversionrate": 0  # Temporär, später berechnet
            }
            # Durchlaufe alle relevanten Status
            for status in ['Termin gebucht', 'Vorquali-Phase', 'Kein Interesse', 'Entscheidungsträger noch nicht erreicht']:
                count = board["status_counts"].get(status, 0)
                change = board["status_changes"].get(status, 0)
                if change > 0:
                    row[status] = f"{count} (+{change})"
                elif change < 0:
                    row[status] = f"{count} ({change})"
                else:
                    row[status] = f"{count}"
            table_rows.append(row)

        df = pd.DataFrame(table_rows)

        # Sicherstellen, dass alle erwarteten Spalten existieren
        expected_columns = ['Termin gebucht', 'Vorquali-Phase', 'Kein Interesse', 'Entscheidungsträger noch nicht erreicht']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = "0"

        # Berechnung der Conversionrate basierend auf den Counts
        def extract_number(val):
            try:
                return int(str(val).split(' ')[0])
            except:
                return 0

        df['Termin gebucht_num'] = df['Termin gebucht'].apply(extract_number)
        df['Vorquali-Phase_num'] = df['Vorquali-Phase'].apply(extract_number)
        df['Kein Interesse_num'] = df['Kein Interesse'].apply(extract_number)

        df['Conversionrate'] = (
            (df['Termin gebucht_num'] + df['Vorquali-Phase_num']) /
            (df['Termin gebucht_num'] + df['Vorquali-Phase_num'] + df['Kein Interesse_num'])
        ).fillna(0) * 100

        # Formatierung der Conversionrate
        df['Conversionrate'] = df['Conversionrate'].map("{:.2f}%".format)

        # Entferne temporäre Spalten
        df.drop(['Termin gebucht_num', 'Vorquali-Phase_num', 'Kein Interesse_num'], axis=1, inplace=True)

        # Reordne die Spalten
        columns_order = ['board_name', 'Conversionrate', 'Termin gebucht', 'Vorquali-Phase', 'Kein Interesse', 'Entscheidungsträger noch nicht erreicht']
        df = df[columns_order]

        # Anzeige der Tabelle im Streamlit-Dashboard
        st.dataframe(df)

        st.markdown("---")
        st.subheader("Detailansicht für ein ausgewähltes Board")

        # Auswahl des Boards für die Detailansicht
        selected_board_name = st.selectbox("Wähle ein Board für Details aus:", df['board_name'])

        # Finde die Daten des ausgewählten Boards
        selected_board = next((item for item in data if item["board_name"] == selected_board_name), None)

        if selected_board:
            st.write(f"### Details für **{selected_board_name}**")

            # Erstelle ein Kreisdiagramm für Statusverteilung (nur bestimmte Status)
            status_counts = selected_board.get("status_counts", {})
            if status_counts:
                # Filtere nur die gewünschten Status
                filtered_status_counts = {k: v for k, v in status_counts.items() if k in ["Termin gebucht", "Vorquali-Phase", "Kein Interesse"]}
                status_df = pd.DataFrame(list(filtered_status_counts.items()), columns=["Status", "Anzahl"])

                fig_status = px.pie(
                    status_df, 
                    names='Status', 
                    values='Anzahl',
                    title='Statusverteilung (Kreisdiagramm)',
                    hole=0.5,
                    color='Status',
                    color_discrete_map=status_colors
                )

                # Hinzufügen schwarzer Ränder zu den Segmenten
                fig_status.update_traces(marker=dict(line=dict(color='black', width=1)))

                st.plotly_chart(fig_status, use_container_width=True, key=f"pie_{selected_board_name}")
            else:
                st.warning("Keine Statusverteilung-Daten vorhanden.")

            # Erstelle ein Balkendiagramm für Statusänderungen (letzte Woche)
            status_changes = selected_board.get("status_changes", {})
            if status_changes:
                # Filtere nur die gewünschten Status
                filtered_status_changes = {k: v for k, v in status_changes.items() if k in ["Termin gebucht", "Vorquali-Phase", "Kein Interesse"]}
                changes_df = pd.DataFrame(list(filtered_status_changes.items()), columns=["Status", "Änderungen"])

                fig_changes = px.bar(
                    changes_df, 
                    x="Status", 
                    y="Änderungen", 
                    title="Statusänderungen (letzte Woche)",
                    color="Status",
                    color_discrete_map=status_colors,
                    text="Änderungen",
                    labels={"Änderungen": "Anzahl der Änderungen"}
                )
                fig_changes.update_traces(textposition='outside')
                fig_changes.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')

                st.plotly_chart(fig_changes, use_container_width=True, key=f"bar_changes_{selected_board_name}")
            else:
                st.warning("Keine Statusänderungen in der letzten Woche.")

            # Bereich für das Linien- oder Flächendiagramm
            st.markdown("---")
            st.subheader("Gebuchte Termine über Zeit")

            termin_gebuucht_dates = selected_board.get("termin_gebuucht_dates", [])
            if termin_gebuucht_dates:
                # Erstelle ein DataFrame aus den Terminen
                df_dates = pd.DataFrame(termin_gebuucht_dates, columns=["TerminDatum"])

                # Konvertiere in datetime, falls nicht bereits
                df_dates['TerminDatum'] = pd.to_datetime(df_dates['TerminDatum'])

                # Entferne Zeitzoneninformationen, falls vorhanden
                if df_dates['TerminDatum'].dt.tz is not None:
                    df_dates['TerminDatum'] = df_dates['TerminDatum'].dt.tz_convert('UTC').dt.tz_localize(None)

                # Setze das Datum als Index
                df_dates.set_index('TerminDatum', inplace=True)

                # Resample nach Tagen und zählen
                df_daily = df_dates.resample('D').size().reset_index(name='Anzahl_Termine')

                # Sortiere die Daten
                df_daily = df_daily.sort_values('TerminDatum').reset_index(drop=True)

                # Berechne die kumulative Summe
                df_daily['Kumulative_Termine'] = df_daily['Anzahl_Termine'].cumsum()

                # Korrekte kumulative Summe sicherstellen
                total_termine = df_daily['Anzahl_Termine'].sum()
                if df_daily['Kumulative_Termine'].iloc[-1] != total_termine:
                    st.warning("Die kumulative Summe stimmt nicht mit der Gesamtanzahl der Termine überein.")
                    # Korrigiere die kumulative Summe
                    df_daily['Kumulative_Termine'] = df_daily['Anzahl_Termine'].cumsum()

                # Erstelle das Linien- oder Flächendiagramm
                fig_area = go.Figure()

                # Farbe für die Linie und das Füllung
                line_color = status_colors.get("Termin gebucht", "#00c875")
                fill_color = hex_to_rgba(line_color, alpha=0.3)

                # Linie für kumulative Termine mit gefülltem Bereich darunter und glatter Kurve
                fig_area.add_trace(go.Scatter(
                    x=df_daily['TerminDatum'],
                    y=df_daily['Kumulative_Termine'],
                    mode='lines',
                    name='Kumulative Termine',
                    line=dict(color=line_color),
                    fill='tozeroy',  # Füllt den Bereich unter der Linie
                    fillcolor=fill_color,  # 30% Deckkraft, gleiche Farbe wie die Linie
                    line_shape='spline',  # Spline für glatte Linien
                    connectgaps=True  # Verbindet Lücken, falls vorhanden
                ))

                fig_area.update_layout(
                    title="Gebuchte Termine über Zeit",
                    xaxis_title="Datum",
                    yaxis_title="Anzahl Termine",
                    xaxis=dict(tickangle=45, tickformat="%Y-%m-%d"),
                    hovermode="x unified"
                )

                st.plotly_chart(fig_area, use_container_width=True, key="line_area_chart")
            else:
                st.warning("Keine gebuchten Termine-Daten vorhanden.")

        else:
            st.error("Keine Daten für das ausgewählte Board gefunden.")

    except Exception as e:
        st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

# Beispiel-URLs
original_urls = [
    "https://view.monday.com/6917933293-38a536695040a37293e93358c7a731dd?r=use1",
    "https://view.monday.com/6265759681-dc9076cc9dd4c09f1d0d59696401024b?r=use1",
    "https://view.monday.com/6265760602-15ff84c644fd114de2a84f58dbbcc9b2?r=use1",
    "https://view.monday.com/6481440664-4720ef13b76d5d34bb5b7324055907f2?r=use1",
    "https://view.monday.com/7741871585-2ebc2d9e8039e4f7f2dd429af1f61c0d?r=use1",
    "https://view.monday.com/7751658853-e248eb0d3f38f213948a21706bb4735d?r=use1",
    "https://view.monday.com/7470906336-d7f6b6df950e2810f7b244a8b1641181?r=use1",
    "https://view.monday.com/7294579207-6c1830474fb76ee1b386c4dc6df5d375?r=use1",
    "https://view.monday.com/6917195148-a4e33cdd18150dbff8239a75e4f0d789?r=use1"
]

# Initialisierung des Session State
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'all_termin_gebuucht_dates' not in st.session_state:
    st.session_state['all_termin_gebuucht_dates'] = []

# Streamlit-Frontend
st.sidebar.title("Einstellungen")
st.sidebar.write("Aktualisierung der Analyse und URL-Verwaltung")
urls_input = st.sidebar.text_area("URLs (eine pro Zeile):", "\n".join(original_urls)).split("\n")

if st.sidebar.button("Analyse starten"):
    with st.spinner("Daten werden abgerufen..."):
        try:
            # Nutzung des bestehenden Event Loops von Streamlit
            data, all_dates = asyncio.run(analyze_multiple_boards(urls_input))
            merged_data, all_termin_gebuucht_dates = merge_similar_boards(data, similarity_threshold=85)
            st.session_state['data'] = merged_data  # Speicherung der Daten
            st.session_state['all_termin_gebuucht_dates'] = all_termin_gebuucht_dates  # Speicherung der Termine
        except Exception as e:
            st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

# Anzeige des Dashboards, falls Daten im Session State vorhanden sind
if st.session_state['data']:
    display_dashboard(st.session_state['data'], st.session_state['all_termin_gebuucht_dates'])
