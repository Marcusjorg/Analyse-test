import asyncio
import aiohttp
from collections import defaultdict, Counter
from rapidfuzz import fuzz, process
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlparse

# Fuzzy Matching Funktion
def fuzzy_grouping(triggers, threshold=60):
    grouped_triggers = {}
    trigger_mapping = {}
    for trigger in triggers:
        match_found = False
        for master_trigger in grouped_triggers.keys():
            if fuzz.ratio(trigger, master_trigger) >= threshold:
                grouped_triggers[master_trigger].append(trigger)
                trigger_mapping[trigger] = master_trigger
                match_found = True
                break
        if not match_found:
            grouped_triggers[trigger] = [trigger]
            trigger_mapping[trigger] = trigger
    return grouped_triggers, trigger_mapping

# Berechnung der Statuswerte inkl. Conversionrate
def calculate_status_values(pulses, status_column_id, status_labels, trigger_mapping, trigger_column_id, board_name):
    trigger_status_counts = defaultdict(lambda: Counter())

    # Relevante Status definieren
    relevant_statuses = {
        "Termin gebucht": "Termin gebucht",
        "Vorquali-Phase": "Vorquali-Phase",
        "Kein Interesse": "Kein Interesse"
    }

    for pulse in pulses:
        column_values = pulse.get("column_values", {})
        original_trigger = column_values.get(trigger_column_id, None)
        if not original_trigger:
            continue

        # Hauptgruppe des Triggers ermitteln
        trigger = trigger_mapping.get(original_trigger, original_trigger)

        status_index = column_values.get(status_column_id, {}).get("index")
        raw_status = status_labels.get(str(status_index), "Unknown")
        status = relevant_statuses.get(raw_status, "Entscheidungsträger noch nicht erreicht")

        trigger_status_counts[trigger][status] += 1

    results = []
    for trigger, counts in trigger_status_counts.items():
        termin_gebucht = counts.get("Termin gebucht", 0)
        kein_interesse = counts.get("Kein Interesse", 0)
        vorquali_phase = counts.get("Vorquali-Phase", 0)
        entscheidung_nicht_erreicht = counts.get("Entscheidungsträger noch nicht erreicht", 0)

        total = termin_gebucht + kein_interesse
        conversion_rate = ((termin_gebucht + vorquali_phase) / total) * 100 if total > 0 else 0

        summe = termin_gebucht + vorquali_phase + kein_interesse + entscheidung_nicht_erreicht

        results.append({
            "Board": board_name,
            "Trigger": trigger,
            "Conversionrate": round(conversion_rate, 2),
            "Termin gebucht": termin_gebucht,
            "Vorquali-Phase": vorquali_phase,
            "Kein Interesse": kein_interesse,
            "Entscheidungsträger noch nicht erreicht": entscheidung_nicht_erreicht,
            "Summe": summe
        })
    return results

def regroup_triggers_after_merge(trigger_data_list):
    if not trigger_data_list:
        return []

    all_triggers = [entry["Trigger"] for entry in trigger_data_list]

    grouped_triggers, trigger_mapping = fuzzy_grouping(all_triggers, threshold=60)

    aggregated_results = {}
    for entry in trigger_data_list:
        original_trigger = entry["Trigger"]
        main_trigger = trigger_mapping.get(original_trigger, original_trigger)

        if main_trigger not in aggregated_results:
            aggregated_results[main_trigger] = {
                "Board": entry["Board"],
                "Trigger": main_trigger,
                "Conversionrate": 0.0,
                "Termin gebucht": 0,
                "Vorquali-Phase": 0,
                "Kein Interesse": 0,
                "Entscheidungsträger noch nicht erreicht": 0,
                "Summe": 0
            }

        aggregated_results[main_trigger]["Termin gebucht"] += entry["Termin gebucht"]
        aggregated_results[main_trigger]["Vorquali-Phase"] += entry["Vorquali-Phase"]
        aggregated_results[main_trigger]["Kein Interesse"] += entry["Kein Interesse"]
        aggregated_results[main_trigger]["Entscheidungsträger noch nicht erreicht"] += entry["Entscheidungsträger noch nicht erreicht"]

    for key, val in aggregated_results.items():
        termin_gebucht = val["Termin gebucht"]
        vorquali_phase = val["Vorquali-Phase"]
        kein_interesse = val["Kein Interesse"]
        entscheidung_nicht_erreicht = val["Entscheidungsträger noch nicht erreicht"]

        total = termin_gebucht + kein_interesse
        conversion_rate = ((termin_gebucht + vorquali_phase) / total) * 100 if total > 0 else 0
        val["Conversionrate"] = round(conversion_rate, 2)
        val["Summe"] = termin_gebucht + vorquali_phase + kein_interesse + entscheidung_nicht_erreicht

    return list(aggregated_results.values())

def get_column_name(col):
    possible_keys = ['title', 'name', 'label', 'text']
    for key in possible_keys:
        if key in col and col[key]:
            return col[key]
    return 'Unknown'

def find_trigger_column(columns, trigger_column_name):
    for col in columns:
        name = get_column_name(col).lower()
        if trigger_column_name.lower() in name:
            return col.get("id")
    return None

def convert_to_api_url(original_url):
    parsed_url = urlparse(original_url)
    path_parts = parsed_url.path.strip('/').split('/')
    if len(path_parts) >= 1:
        board_id = path_parts[-1].split('?')[0]
        api_url = f"https://view.monday.com/board_data/{board_id}.json"
        return api_url
    else:
        raise ValueError(f"Ungültiges URL-Format: {original_url}")

def hex_to_rgba(hex_color, alpha=0.3):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    if lv == 6:
        r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    elif lv == 3:
        r, g, b = hex_color[0]*2, hex_color[1]*2, hex_color[2]*2
    else:
        raise ValueError("Invalid hex color format")
    return f'rgba({int(r,16)}, {int(g,16)}, {int(b,16)}, {alpha})'

async def analyze_board(session, board, time_threshold):
    try:
        formatted_url = convert_to_api_url(board["url"])
        async with session.get(formatted_url) as response:
            if response.status == 200:
                board_data = await response.json()

                board_name = board_data.get('name', f"Unbekanntes Board ({formatted_url})")

                columns = board_data.get('board_data', {}).get('columns', [])
                status_column = next(
                    (col for col in columns if col.get('type') == 'color' and 'labels' in col), None
                )

                if not status_column:
                    st.warning(f"Keine Status-Spalte im Board '{board_name}' gefunden.")
                    return {
                        "board_name": board_name,
                        "status_counts": {},
                        "status_changes": {},
                        "termin_gebucht_dates": [],
                        "trigger_data": []
                    }

                status_labels = status_column.get('labels', {})
                pulses = board_data.get('board_data', {}).get('pulses', [])
                status_column_id = status_column.get('id')

                trigger_column_name = board.get("trigger_column_name", "Trigger")
                trigger_column_id = find_trigger_column(columns, trigger_column_name)

                if not trigger_column_id:
                    st.warning(f"Keine Trigger-Spalte mit dem Namen '{trigger_column_name}' im Board '{board_name}' gefunden.")
                    st.info(f"Verfügbare Spalten in '{board_name}':")
                    for col in columns:
                        name = get_column_name(col)
                        st.write(f" - {name} (ID: {col.get('id')})")
                    return {
                        "board_name": board_name,
                        "status_counts": {},
                        "status_changes": {},
                        "termin_gebucht_dates": [],
                        "trigger_data": []
                    }

                status_counts = Counter()
                status_changes = Counter()
                termin_gebucht_dates = []

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
                        status = relevant_statuses.get(raw_status, "Entscheidungsträger noch nicht erreicht")
                        status_counts[status] += 1

                        if changed_at:
                            try:
                                changed_at_dt = datetime.fromisoformat(changed_at.replace("Z", "+00:00"))
                                changed_at_dt = changed_at_dt.replace(tzinfo=None)
                                if status == "Termin gebucht":
                                    termin_gebucht_dates.append(changed_at_dt)
                                if changed_at_dt >= time_threshold:
                                    status_changes[status] += 1
                            except ValueError:
                                st.error(f"Ungültiges Datum im Board '{board_name}': {changed_at}")

                trigger_data = []
                if trigger_column_id:
                    raw_triggers = [pulse.get("column_values", {}).get(trigger_column_id, "") for pulse in pulses]
                    raw_triggers = [trigger for trigger in raw_triggers if trigger]

                    if raw_triggers:
                        grouped_triggers, trigger_mapping = fuzzy_grouping(raw_triggers)
                        trigger_results = calculate_status_values(
                            pulses, status_column_id, status_labels, trigger_mapping, trigger_column_id, board_name
                        )
                        trigger_data.extend(trigger_results)

                return {
                    "board_name": board_name,
                    "status_counts": dict(status_counts),
                    "status_changes": dict(status_changes),
                    "termin_gebucht_dates": termin_gebucht_dates,
                    "trigger_data": trigger_data
                }
            else:
                st.error(f"Fehler beim Abrufen des Boards: {board['url']}, Status Code: {response.status}")
                return {
                    "board_name": f"Unbekanntes Board ({formatted_url})",
                    "status_counts": {},
                    "status_changes": {},
                    "termin_gebucht_dates": [],
                    "trigger_data": []
                }
    except Exception as e:
        st.error(f"Fehler bei der Analyse des Boards: {board['url']}, Fehler: {e}")
        return {
            "board_name": f"Unbekanntes Board ({board['url']})",
            "status_counts": {},
            "status_changes": {},
            "termin_gebucht_dates": [],
            "trigger_data": []
        }

def get_start_of_current_week():
    today = datetime.now(timezone.utc)
    start_of_week = today - timedelta(days=today.weekday()) 
    return datetime.combine(start_of_week, datetime.min.time())

async def analyze_multiple_boards(boards):
    time_threshold = get_start_of_current_week()
    async with aiohttp.ClientSession() as session:
        tasks = [analyze_board(session, board, time_threshold) for board in boards]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    dashboard_data = []
    all_termin_gebucht_dates = []
    all_trigger_data = defaultdict(list)

    for result in results:
        if isinstance(result, Exception):
            continue
        board_name = result.get("board_name", "Unbekanntes Board")
        counts = result.get("status_counts", {}) or {}
        changes = result.get("status_changes", {}) or {}
        dates = result.get("termin_gebucht_dates", []) or []
        triggers = result.get("trigger_data", []) or {}
        dashboard_data.append({
            "board_name": board_name,
            "status_counts": counts,
            "status_changes": changes,
            "termin_gebucht_dates": dates,
            "trigger_data": triggers
        })
        all_termin_gebucht_dates.extend(dates)
        for trigger_entry in triggers:
            all_trigger_data[board_name].append(trigger_entry)

    return dashboard_data, all_termin_gebucht_dates, all_trigger_data

def merge_similar_boards(data, similarity_threshold=85):
    merged_data = []
    processed = set()
    all_termin_gebucht_dates = []
    all_trigger_data = defaultdict(list)

    for board in data:
        if board["board_name"] in processed:
            continue

        matches = process.extract(
            board["board_name"],
            [b["board_name"] for b in data],
            scorer=fuzz.ratio,
        )

        similar_boards = [
            match[0] for match in matches if match[1] >= similarity_threshold
        ]

        merged_board = {
            "board_name": board["board_name"],
            "status_counts": Counter(),
            "status_changes": Counter(),
            "termin_gebucht_dates": [],
            "trigger_data": []
        }
        for b_name in similar_boards:
            matching_board = next((b for b in data if b["board_name"] == b_name), None)
            if matching_board:
                merged_board["status_counts"].update(matching_board["status_counts"])
                merged_board["status_changes"].update(matching_board["status_changes"])
                merged_board["termin_gebucht_dates"].extend(matching_board["termin_gebucht_dates"])
                merged_board["trigger_data"].extend(matching_board["trigger_data"])
                processed.add(b_name)

        merged_board["trigger_data"] = regroup_triggers_after_merge(merged_board["trigger_data"])

        merged_data.append(merged_board)
        all_termin_gebucht_dates.extend(merged_board["termin_gebucht_dates"])
        for trigger_entry in merged_board["trigger_data"]:
            all_trigger_data[merged_board["board_name"]].append(trigger_entry)

    for board in merged_data:
        board["status_counts"] = dict(board["status_counts"])
        board["status_changes"] = dict(board["status_changes"])

    return merged_data, all_termin_gebucht_dates, all_trigger_data

def display_dashboard(data, all_termin_gebucht_dates, all_trigger_data):
    try:
        st.title("Datenanalyse Dashboard")
        st.write("Tabellarische Übersicht der Boards:")

        table_rows = []
        for board in data:
            row = {
                "board_name": board["board_name"],
                "Conversionrate": 0 
            }
            for status in ['Termin gebucht', 'Vorquali-Phase', 'Kein Interesse', 'Entscheidungsträger noch nicht erreicht']:
                count = board["status_counts"].get(status, 0)
                change = board["status_changes"].get(status, 0)
                if change > 0:
                    row[status] = f"{count} (+{change})"
                elif change < 0:
                    row[status] = f"{count} ({change})"
                else:
                    row[status] = f"{count}"

            termin_gebucht_num = board["status_counts"].get("Termin gebucht", 0)
            vorquali_phase_num = board["status_counts"].get("Vorquali-Phase", 0)
            kein_interesse_num = board["status_counts"].get("Kein Interesse", 0)
            total = termin_gebucht_num + vorquali_phase_num + kein_interesse_num
            conversion_rate = ((termin_gebucht_num + vorquali_phase_num) / total) * 100 if total > 0 else 0
            row["Conversionrate"] = f"{conversion_rate:.2f}%"

            table_rows.append(row)

        df = pd.DataFrame(table_rows)

        expected_columns = ['Termin gebucht', 'Vorquali-Phase', 'Kein Interesse', 'Entscheidungsträger noch nicht erreicht']
        for col in expected_columns:
            if col not in df.columns:
                df[col] = "0"

        columns_order = ['board_name', 'Conversionrate', 'Termin gebucht', 'Vorquali-Phase', 'Kein Interesse', 'Entscheidungsträger noch nicht erreicht']
        df = df[columns_order]

        st.dataframe(df)

        st.markdown("---")
        st.subheader("Detailansicht für ein ausgewähltes Board")

        selected_board_name = st.selectbox("Wähle ein Board für Details aus:", df['board_name'])

        selected_board = next((item for item in data if item["board_name"] == selected_board_name), None)

        if selected_board:
            st.write(f"### Details für **{selected_board_name}**")

            status_counts = selected_board.get("status_counts", {})
            if status_counts:
                filtered_status_counts = {k: v for k, v in status_counts.items() if k in ["Termin gebucht", "Vorquali-Phase", "Kein Interesse"]}
                status_df = pd.DataFrame(list(filtered_status_counts.items()), columns=["Status", "Anzahl"])

                status_colors = {
                    "Termin gebucht": "#00c875",
                    "Vorquali-Phase": "#579bfc",
                    "Kein Interesse": "#c4c4c4",
                    "Entscheidungsträger noch nicht erreicht": "#ffcc00"
                }

                fig_status = px.pie(
                    status_df, 
                    names='Status', 
                    values='Anzahl',
                    title='Statusverteilung (Kreisdiagramm)',
                    hole=0.5,
                    color='Status',
                    color_discrete_map=status_colors
                )

                fig_status.update_traces(marker=dict(line=dict(color='black', width=1)))

                st.plotly_chart(fig_status, use_container_width=True, key=f"pie_{selected_board_name}")
            else:
                st.warning("Keine Statusverteilung-Daten vorhanden.")

            status_changes = selected_board.get("status_changes", {})
            if status_changes:
                filtered_status_changes = {k: v for k, v in status_changes.items() if k in ["Termin gebucht", "Vorquali-Phase", "Kein Interesse"]}
                if filtered_status_changes:
                    changes_df = pd.DataFrame(list(filtered_status_changes.items()), columns=["Status", "Änderungen"])

                    status_colors = {
                        "Termin gebucht": "#00c875",
                        "Vorquali-Phase": "#579bfc",
                        "Kein Interesse": "#c4c4c4",
                        "Entscheidungsträger noch nicht erreicht": "#ffcc00"
                    }

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
                    st.warning("Keine relevanten Statusänderungen in der letzten Woche.")
            else:
                st.warning("Keine Statusänderungen in der letzten Woche.")

            st.markdown("---")
            st.subheader("Gebuchte Termine (wöchentlich)")

            termin_gebucht_dates = selected_board.get("termin_gebucht_dates", [])
            if termin_gebucht_dates:
                df_dates = pd.DataFrame(termin_gebucht_dates, columns=["TerminDatum"])
                df_dates['TerminDatum'] = pd.to_datetime(df_dates['TerminDatum'])
                
                # Du könntest hier zusätzlich eine Zeitzone setzen, falls gewünscht:
                # df_dates['TerminDatum'] = df_dates['TerminDatum'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')

                df_dates.set_index('TerminDatum', inplace=True)

                # Wöchentliche Auswertung mit Label auf Montag:
                df_weekly = (
                    df_dates
                    .resample('W-MON', label='left', closed='left')
                    .size()
                    .reset_index(name='Anzahl_Termine')
                )
                df_weekly = df_weekly.sort_values('TerminDatum').reset_index(drop=True)
                df_weekly['Kumulative_Termine'] = df_weekly['Anzahl_Termine'].cumsum()

                # Toggle zwischen "Kumulativ" und "Pro Woche"
                darstellung = st.radio(
                    "Darstellung:",
                    ["Kumulativ", "Pro Woche"],
                    horizontal=True
                )

                # Falls "Kumulativ" ausgewählt ist, hänge eine Dummy-Zeile an,
                # damit die Kurve nicht direkt > 0 startet
                if darstellung == "Kumulativ":
                    if not df_weekly.empty:
                        first_row = df_weekly.iloc[0].copy()
                        first_date = first_row['TerminDatum']
                        dummy_date = first_date - pd.Timedelta(days=7)
                        new_row = {
                            'TerminDatum': dummy_date,
                            'Anzahl_Termine': 0,
                            'Kumulative_Termine': 0
                        }
                        df_weekly = pd.concat([pd.DataFrame([new_row]), df_weekly], ignore_index=True)

                    y_value = "Kumulative_Termine"
                    chart_title = "Gebuchte Termine (kumulativ pro KW)"
                else:
                    y_value = "Anzahl_Termine"
                    chart_title = "Gebuchte Termine (nur diese KW)"

                c1, c2 = st.columns([2,1])  # linke Spalte (2/3) für Diagramm, rechte Spalte (1/3) für Tabelle

                with c1:
                    fig_weekly = go.Figure()
                    weekly_line_color = "#00c875"
                    weekly_fill_color = hex_to_rgba(weekly_line_color, alpha=0.3)

                    fig_weekly.add_trace(go.Scatter(
                        x=df_weekly['TerminDatum'],
                        y=df_weekly[y_value],
                        mode='lines+markers',
                        name='Termine',
                        line=dict(color=weekly_line_color),
                        fill='tozeroy',
                        fillcolor=weekly_fill_color,
                        line_shape='spline'
                    ))

                    fig_weekly.update_layout(
                        title=chart_title,
                        xaxis_title="Montag (Start der KW)",
                        yaxis_title="Anzahl Termine",
                        xaxis=dict(tickangle=45),
                        hovermode="x unified"
                    )

                    st.plotly_chart(fig_weekly, use_container_width=True)

                with c2:
                    st.write("### Gebuchte Termine pro KW")

                    df_weekly_display = df_weekly.copy()
                    # Neue Spalte mit "KW X"
                    df_weekly_display['KW'] = "KW " + df_weekly_display['TerminDatum'].dt.isocalendar().week.astype(str)

                    # Zeige abhängig von 'darstellung' entweder kumulative oder einzelne Werte an
                    if darstellung == "Kumulativ":
                        df_weekly_display = df_weekly_display[['KW', 'Kumulative_Termine']]
                        df_weekly_display.rename(
                            columns={'Kumulative_Termine': 'Termine (kumulativ)'}, 
                            inplace=True
                        )
                    else:
                        df_weekly_display = df_weekly_display[['KW', 'Anzahl_Termine']]
                        df_weekly_display.rename(
                            columns={'Anzahl_Termine': 'Termine (pro KW)'}, 
                            inplace=True
                        )

                    st.dataframe(df_weekly_display)

            else:
                st.warning("Keine gebuchten Termine-Daten vorhanden.")

            st.markdown("---")
            st.subheader("Trigger Events Übersicht")

            trigger_entries = all_trigger_data.get(selected_board_name, [])
            if trigger_entries:
                trigger_df = pd.DataFrame(trigger_entries)

                trigger_columns_order = ["Trigger", "Summe", "Conversionrate", "Termin gebucht", "Vorquali-Phase", "Kein Interesse", "Entscheidungsträger noch nicht erreicht"]
                for col in trigger_columns_order:
                    if col not in trigger_df.columns:
                        trigger_df[col] = 0
                trigger_df = trigger_df[trigger_columns_order]

                st.write("### Trigger Events Tabelle:")
                st.dataframe(trigger_df)
            else:
                st.info("Keine Trigger-Daten für dieses Board verfügbar.")

    except Exception as e:
        st.error(f"Fehler bei der Anzeige des Dashboards: {e}")


# --- Session State Handling ---
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'all_termin_gebucht_dates' not in st.session_state:
    st.session_state['all_termin_gebucht_dates'] = []
if 'all_trigger_data' not in st.session_state:
    st.session_state['all_trigger_data'] = {}

# --- Sidebar-Einstellungen ---
st.sidebar.title("Einstellungen")
st.sidebar.write("Wählen Sie die Kategorie und geben Sie die Board-URLs ein.")

category = st.sidebar.radio("Kategorie auswählen:", ["Aktuelle Kunden", "Ex-Kunden"])

if category == "Aktuelle Kunden":
    default_current = """https://view.monday.com/6917933293-38a536695040a37293e93358c7a731dd?r=use1
https://view.monday.com/6265759681-dc9076cc9dd4c09f1d0d59696401024b?r=use1
https://view.monday.com/6265760602-15ff84c644fd114de2a84f58dbbcc9b2?r=use1
https://view.monday.com/6481440664-4720ef13b76d5d34bb5b7324055907f2?r=use1
https://view.monday.com/7741871585-2ebc2d9e8039e4f7f2dd429af1f61c0d?r=use1
https://view.monday.com/7751658853-e248eb0d3f38f213948a21706bb4735d?r=use1
https://view.monday.com/7470906336-d7f6b6df950e2810f7b244a8b1641181?r=use1
https://view.monday.com/7294579207-6c1830474fb76ee1b386c4dc6df5d375?r=use1
https://view.monday.com/6917195148-a4e33cdd18150dbff8239a75e4f0d789?r=use1
"""
    current_customer_urls = st.sidebar.text_area("Geben Sie die Board-URLs ein (eine pro Zeile):", value=default_current)
else:
    default_ex = """https://view.monday.com/6369200775-3ac90ec0c4d54166fb3728364dbff042?r=use1
https://view.monday.com/6548399740-68e4f8e695d3fc9457ee33b890c85ff8?r=use1
https://view.monday.com/6369286986-bc2cbbc9ec536a1c6f213880d271ae42?r=use1
https://view.monday.com/7323105057-111f756b151e851966d4219d05d9b5e6?r=use1
https://view.monday.com/6471576721-babe4fbb94f25aa86971139719ef3db8?r=use1
https://view.monday.com/7032932463-6ab5ec8e812c040c851656d8c7f65e28?r=use1
https://view.monday.com/7032934368-9a5518176d19d1847915c434544c7640?r=use1
https://view.monday.com/5948228574-383e022887aaa0f7dbd2eaf01681cddd?r=use1
https://view.monday.com/6462737612-464a5262bc14a2ac3421a779b20f0463?r=use1
https://view.monday.com/5719544683-1a1e791364873414b8b0e189bee402b6?r=use1
https://view.monday.com/5035215808-3a3f16c78f35eca13f1ce74798ccc078?r=use1
https://view.monday.com/6369152758-a25c3a1683b404a355c285e63ce082bd?r=use1
https://view.monday.com/6265753631-6b0d4729c34ded450e9edd059380a8b2?r=use1
https://view.monday.com/6265755698-1aa953ec59122eed6b78463f2077cb15?r=use1
https://view.monday.com/5855732867-98e73d205b0316c240f21e5564860f4b?r=use1
https://view.monday.com/7741863679-9baeb49c20c15039d7acff12fdfc5856?r=use1
https://view.monday.com/6462730033-dcdacc7f624a0fafdfad2558011b1e45?r=use1
https://view.monday.com/7067633860-4b9339b5fe4f4957f4ac3c6991769395?r=use1
"""
    ex_customer_urls = st.sidebar.text_area("Geben Sie die Board-URLs ein (eine pro Zeile):", value=default_ex)

boards = []
if category == "Aktuelle Kunden" and current_customer_urls.strip():
    urls = [u.strip() for u in current_customer_urls.split('\n') if u.strip()]
    for i, url in enumerate(urls, start=1):
        boards.append({
            "name": f"Aktueller Kunde {i}",
            "url": url,
            "trigger_column_name": "Trigger"
        })
elif category == "Ex-Kunden" and ex_customer_urls.strip():
    urls = [u.strip() for u in ex_customer_urls.split('\n') if u.strip()]
    for i, url in enumerate(urls, start=1):
        boards.append({
            "name": f"Ex-Kunde {i}",
            "url": url,
            "trigger_column_name": "Trigger"
        })

# --- Button zum Starten der Analyse ---
if st.sidebar.button("Analyse starten"):
    if boards:
        with st.spinner("Daten werden abgerufen..."):
            try:
                data, all_termin_gebucht_dates, all_trigger_data = asyncio.run(analyze_multiple_boards(boards))
                merged_data, merged_termin_gebucht_dates, merged_trigger_data = merge_similar_boards(data, similarity_threshold=85)
                st.session_state['data'] = merged_data
                st.session_state['all_termin_gebucht_dates'] = merged_termin_gebucht_dates
                st.session_state['all_trigger_data'] = merged_trigger_data
            except Exception as e:
                st.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
    else:
        st.error("Bitte geben Sie mindestens eine gültige Board-URL an.")

# --- Dashboard anzeigen, falls Daten vorhanden ---
if st.session_state['data']:
    display_dashboard(st.session_state['data'], st.session_state['all_termin_gebucht_dates'], st.session_state['all_trigger_data'])
