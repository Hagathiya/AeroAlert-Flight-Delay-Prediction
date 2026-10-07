"""
analytics.py
Generates the dashboard charts from the flights dataset.

Run this ONCE (python analytics.py) and it writes PNG charts into
static/charts/ plus a stats.json summary that the dashboard reads.
Doing it offline keeps the web app fast - no heavy pandas work per request.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt
import pandas as pd

DATASET_PATH = "dataset/flights.csv"
CHART_DIR = "static/charts"
STATS_PATH = "static/charts/stats.json"
ROWS = 100000  # same subset used for training

# Visual style - muted, readable, consistent across charts
PLOT_BG = "#ffffff"
BAR_COLOR = "#2563eb"
ACCENT_COLOR = "#dc2626"
GRID_COLOR = "#e5e7eb"
TEXT_COLOR = "#1f2937"


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, color=TEXT_COLOR, pad=14, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_COLOR)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)


def save(fig, name):
    path = os.path.join(CHART_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=PLOT_BG)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    os.makedirs(CHART_DIR, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, nrows=ROWS, low_memory=False)
    print(f"  loaded {len(df):,} rows")

    # Build the same target used in training
    df = df.dropna(subset=["ARR_DELAY"])
    df["DELAYED"] = (df["ARR_DELAY"] > 15).astype(int)

    # Derive hour and weekday
    df["DEP_HOUR"] = (df["CRS_DEP_TIME"] // 100).astype(int)
    df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    df["DAY_OF_WEEK"] = df["FL_DATE"].dt.dayofweek
    df["MONTH"] = df["FL_DATE"].dt.month

    stats = {}
    stats["total_flights"] = int(len(df))
    stats["delayed_flights"] = int(df["DELAYED"].sum())
    stats["delay_rate"] = round(float(df["DELAYED"].mean()) * 100, 1)
    stats["avg_delay_minutes"] = round(
        float(df.loc[df["DELAYED"] == 1, "ARR_DELAY"].mean()), 1
    )
    stats["airports_covered"] = int(df["ORIGIN"].nunique())
    stats["airlines_covered"] = int(df["AIRLINE_CODE"].nunique())

    print("Building charts...")

    # 1. Delay rate by departure hour
    by_hour = df.groupby("DEP_HOUR")["DELAYED"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=PLOT_BG)
    colors = [ACCENT_COLOR if v == by_hour.max() else BAR_COLOR for v in by_hour.values]
    ax.bar(by_hour.index, by_hour.values, color=colors, width=0.75)
    style_axes(
        ax,
        "Delay rate rises through the day",
        "Scheduled departure hour",
        "Flights delayed (%)",
    )
    ax.set_xticks(list(by_hour.index))
    ax.set_xticklabels([f"{h:02d}" for h in by_hour.index], fontsize=8)
    save(fig, "delay_by_hour.png")
    stats["worst_hour"] = int(by_hour.idxmax())
    stats["worst_hour_rate"] = round(float(by_hour.max()), 1)
    stats["best_hour"] = int(by_hour.idxmin())
    stats["best_hour_rate"] = round(float(by_hour.min()), 1)

    # 2. Delay rate by airline (top 12 by volume)
    top_airlines = df["AIRLINE_CODE"].value_counts().head(12).index
    by_airline = (
        df[df["AIRLINE_CODE"].isin(top_airlines)]
        .groupby("AIRLINE_CODE")["DELAYED"]
        .mean()
        .sort_values()
        * 100
    )
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=PLOT_BG)
    colors = [ACCENT_COLOR if v == by_airline.max() else BAR_COLOR for v in by_airline.values]
    ax.barh(by_airline.index, by_airline.values, color=colors, height=0.72)
    style_axes(ax, "Delay rate varies widely by airline", "Flights delayed (%)", "")
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    save(fig, "delay_by_airline.png")
    stats["worst_airline"] = str(by_airline.idxmax())
    stats["worst_airline_rate"] = round(float(by_airline.max()), 1)
    stats["best_airline"] = str(by_airline.idxmin())
    stats["best_airline_rate"] = round(float(by_airline.min()), 1)

    # 3. Delay rate by day of week
    by_day = df.groupby("DAY_OF_WEEK")["DELAYED"].mean() * 100
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=PLOT_BG)
    colors = [ACCENT_COLOR if v == by_day.max() else BAR_COLOR for v in by_day.values]
    ax.bar([day_names[i] for i in by_day.index], by_day.values, color=colors, width=0.6)
    style_axes(ax, "Delay rate by day of week", "", "Flights delayed (%)")
    save(fig, "delay_by_day.png")

    # 4. Delay rate by distance band
    bands = [0, 300, 600, 1000, 1500, 2500, 10000]
    labels = ["<300", "300-600", "600-1k", "1k-1.5k", "1.5k-2.5k", "2.5k+"]
    df["DIST_BAND"] = pd.cut(df["DISTANCE"], bins=bands, labels=labels)
    by_dist = df.groupby("DIST_BAND", observed=True)["DELAYED"].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 3.2), facecolor=PLOT_BG)
    ax.bar(by_dist.index.astype(str), by_dist.values, color=BAR_COLOR, width=0.6)
    style_axes(ax, "Delay rate by flight distance", "Distance (miles)", "Flights delayed (%)")
    save(fig, "delay_by_distance.png")

    # 5. Overall split
    fig, ax = plt.subplots(figsize=(4.5, 3.4), facecolor=PLOT_BG)
    counts = [len(df) - df["DELAYED"].sum(), df["DELAYED"].sum()]
    ax.bar(["On time", "Delayed"], counts, color=[BAR_COLOR, ACCENT_COLOR], width=0.5)
    style_axes(ax, "Class balance in training data", "", "Flights")
    save(fig, "class_balance.png")

    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote {STATS_PATH}")

    print("\nSummary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nDone. Charts are in static/charts/")


if __name__ == "__main__":
    main()