import asyncio
import aiohttp
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from rapidfuzz import process, fuzz

# Funktion, um die URL zu formatieren
def format_url(original_url):
    base_url = "https://view.monday.com/board_data/"
    url_core = original_url.split('?')[0]
    formatted_url = f"{base_url}{url_core.split('/')[-1]}.json"
    return formatted_url

# Funktion, um ein einzelnes Board zu analysieren
async def analyze_board(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                board_data = await response.json()

                # Boardnamen extrahieren
                board_name = board_data.get('name', 'Unbekanntes Board')

                # Spalteninformationen extrahieren
                columns = board_data['board_data']['columns']

                # Dynamische Erkennung der Status-Spalte
                status_column = next(
                    (col for col in columns if col.get('type') == 'color' and 'labels' in col), None
                )

                if not status_column:
                    return {"board_name": board_name, "status_counts": None}

                # Labels aus der erkannten Status-Spalte extrahieren
                status_labels = status_column.get('labels', {})

                # Pulses extrahieren
                pulses = board_data['board_data'].get('pulses', [])

                # Dynamische Spalten-ID der Statuswerte
                status_column_id = status_column.get('id')

                # Statuscodes aus den Pulses extrahieren
                status_codes = [
                    pulse['column_values'].get(status_column_id, {}).get('index')
                    for pulse in pulses
                    if status_column_id in pulse['column_values']
                ]

                # Statuscodes zu Labels übersetzen
                translated_statuses = [status_labels.get(str(code), "Unknown") for code in status_codes]

                # Gruppierungen definieren
                def group_status(status):
                    if status in {"Vorquali-Phase", "Termin gebucht", "Kein Interesse"}:
                        return status
                    else:
                        return "Entscheidungsträger noch nicht erreicht"

                # Gruppierte Statuswerte
                grouped_statuses = [group_status(status) for status in translated_statuses]

                # Häufigkeiten der gruppierten Statuswerte zählen
                status_counts = Counter(grouped_statuses)
                return {"board_name": board_name, "status_counts": status_counts}
            else:
                return {"board_name": f"Unbekanntes Board ({url})", "status_counts": None}
    except Exception as e:
        return {"board_name": f"Unbekanntes Board ({url})", "status_counts": None}

# Hauptfunktion zur Analyse mehrerer Boards
async def analyze_multiple_boards(original_urls):
    formatted_urls = [format_url(url) for url in original_urls]
    async with aiohttp.ClientSession() as session:
        tasks = [analyze_board(session, url) for url in formatted_urls]
        results = await asyncio.gather(*tasks)

    dashboard_data = []
    for result in results:
        board_name = result["board_name"]
        counts = result["status_counts"]
        if counts:
            dashboard_data.append({"board_name": board_name, **counts})
        else:
            dashboard_data.append({"board_name": board_name})
    return dashboard_data

# Funktion zur Zusammenführung ähnlicher Namen
def merge_similar_boards(data, similarity_threshold=85):
    merged_data = []
    processed = set()

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
        merged_board = {"board_name": board["board_name"]}
        for b_name in similar_boards:
            matching_board = next(b for b in data if b["board_name"] == b_name)
            for key, value in matching_board.items():
                if key != "board_name":
                    merged_board[key] = merged_board.get(key, 0) + value
            processed.add(b_name)

        merged_data.append(merged_board)

    return merged_data

# Funktion zur Anzeige des Dashboards
def display_dashboard(data):
    df = pd.DataFrame(data).fillna(0)

    # Conversionrate berechnen
    df['Conversionrate'] = (
        (df['Termin gebucht'] + df['Vorquali-Phase']) /
        (df['Termin gebucht'] + df['Vorquali-Phase'] + df['Kein Interesse'])
    ).fillna(0) * 100

    # Neuordnung der Spalten
    columns_order = ['board_name', 'Conversionrate', 'Termin gebucht', 'Vorquali-Phase', 'Kein Interesse', 'Entscheidungsträger noch nicht erreicht']
    df = df[columns_order]

    # Tabellarische Übersicht anzeigen
    st.title("Datenanalyse Dashboard")
    st.write("Tabellarische Übersicht der Boards:")
    st.dataframe(df)

    # Diagramme für jedes Board erstellen
    for _, row in df.iterrows():
        board_name = row['board_name']
        values = row[2:].values  # Diagramm zeigt nur Statusarten
        labels = row.index[2:]

        st.subheader(f"Statusverteilung für {board_name}")
        st.bar_chart(pd.DataFrame(values, index=labels, columns=["Anzahl"]))

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

# Streamlit-Frontend
st.sidebar.title("Einstellungen")
st.sidebar.write("Aktualisierung der Analyse und URL-Verwaltung")
urls_input = st.sidebar.text_area("URLs (eine pro Zeile):", "\n".join(original_urls)).split("\n")

if st.sidebar.button("Analyse starten"):
    with st.spinner("Daten werden abgerufen..."):
        data = asyncio.run(analyze_multiple_boards(urls_input))
        merged_data = merge_similar_boards(data)
        display_dashboard(merged_data)
