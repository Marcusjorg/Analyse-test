import asyncio
import aiohttp
from collections import Counter
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from datetime import datetime, timedelta, timezone

# [Alle oben definierten Funktionen hier einfügen: format_url, analyze_board, get_start_of_current_week, analyze_multiple_boards, merge_similar_boards, display_dashboard]
import asyncio
import aiohttp
from collections import Counter
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz
from datetime import datetime, timedelta, timezone

# Funktion, um die URL zu formatieren
def format_url(original_url):
    base_url = "https://view.monday.com/board_data/"
    url_core = original_url.split('?')[0]
    formatted_url = f"{base_url}{url_core.split('/')[-1]}.json"
    return formatted_url

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
                    return {"board_name": board_name, "status_counts": {}, "status_changes": {}}

                # Labels aus der erkannten Status-Spalte extrahieren
                status_labels = status_column.get('labels', {})

                # Pulses extrahieren
                pulses = board_data.get('board_data', {}).get('pulses', [])

                # Dynamische Spalten-ID der Statuswerte
                status_column_id = status_column.get('id')

                # Statuscodes und Änderungsdaten extrahieren
                status_counts = Counter()
                status_changes = Counter()

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
                                if changed_at_dt >= time_threshold:
                                    status_changes[status] += 1
                            except ValueError:
                                st.error(f"Ungültiges Datum im Board {board_name}: {changed_at}")

                return {
                    "board_name": board_name,
                    "status_counts": dict(status_counts),
                    "status_changes": dict(status_changes)
                }
            else:
                return {"board_name": f"Unbekanntes Board ({formatted_url})", "status_counts": {}, "status_changes": {}}
    except Exception as e:
        st.error(f"Fehler bei der Analyse des Boards: {url}, Fehler: {e}")
        return {"board_name": f"Unbekanntes Board ({url})", "status_counts": {}, "status_changes": {}}

# Hauptfunktion zur Analyse mehrerer Boards
def get_start_of_current_week():
    today = datetime.now(timezone.utc)
    start_of_week = today - timedelta(days=today.weekday())  # Montag dieser Woche
    return datetime.combine(start_of_week, datetime.min.time(), tzinfo=timezone.utc)

async def analyze_multiple_boards(original_urls):
    time_threshold = get_start_of_current_week()  # Beginn der aktuellen Woche
    async with aiohttp.ClientSession() as session:
        tasks = [analyze_board(session, url, time_threshold) for url in original_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    dashboard_data = []
    for result in results:
        if isinstance(result, Exception):
            continue
        board_name = result.get("board_name", "Unbekanntes Board")
        counts = result.get("status_counts", {}) or {}
        changes = result.get("status_changes", {}) or {}
        dashboard_data.append({
            "board_name": board_name,
            "status_counts": counts,
            "status_changes": changes
        })
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
        merged_board = {
            "board_name": board["board_name"],
            "status_counts": Counter(),
            "status_changes": Counter()
        }
        for b_name in similar_boards:
            matching_board = next(b for b in data if b["board_name"] == b_name)
            merged_board["status_counts"].update(matching_board["status_counts"])
            merged_board["status_changes"].update(matching_board["status_changes"])
            processed.add(b_name)

        merged_data.append(merged_board)

    # Konvertiere Counters zurück zu normalen Dictionaries
    for board in merged_data:
        board["status_counts"] = dict(board["status_counts"])
        board["status_changes"] = dict(board["status_changes"])

    return merged_data

# Funktion zur Anzeige des Dashboards
def display_dashboard(data):
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

    # Funktion zur Farbgebung der Conversionrate
    def colorize_conversion_rate(val):
        try:
            num = float(val.strip('%'))
            if num > 20:
                color = 'green'
            elif 10 <= num <= 20:
                color = 'orange'
            else:
                color = 'red'
            return color
        except:
            return 'black'

    df['Conversionrate_color'] = df['Conversionrate'].apply(colorize_conversion_rate)

    # Entferne temporäre Spalten
    df.drop(['Termin gebucht_num', 'Vorquali-Phase_num', 'Kein Interesse_num'], axis=1, inplace=True)

    # Reordne die Spalten
    columns_order = ['board_name', 'Conversionrate', 'Termin gebucht', 'Vorquali-Phase', 'Kein Interesse', 'Entscheidungsträger noch nicht erreicht']
    df = df[columns_order]

    # Anwendung von Styling mit Pandas
    def highlight_conversion(val):
        color = 'black'
        if isinstance(val, str):
            num = float(val.strip('%'))
            if num > 20:
                color = 'green'
            elif 10 <= num <= 20:
                color = 'orange'
            else:
                color = 'red'
        return f'color: {color}'

    styled_df = df.style.applymap(highlight_conversion, subset=['Conversionrate'])

    # Anpassung der Tabellenbreite und Höhe via CSS
    css = """
    <style>
    .dataframe tbody tr {
        height: 50px;
    }
    .dataframe thead th {
        background-color: #f2f2f2;
        text-align: left;
    }
    </style>
    """

    # Anzeige der Tabelle im Streamlit-Dashboard
    st.title("Datenanalyse Dashboard")
    st.write("Tabellarische Übersicht der Boards:")

    # Kombiniere CSS mit der Tabelle
    st.markdown(css, unsafe_allow_html=True)
    st.dataframe(styled_df)

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
